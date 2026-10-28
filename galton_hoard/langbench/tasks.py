"""The 27 tasks of the language bench, each written once and localised to Spanish and English with the same id.

A task is declared with its instructions in both languages (Translate-Free), the entities it mentions as ``{kind:key}`` placeholders that read
the canonical map (Translate-Canonical), the tool calls of its oracle per language, and one language-neutral set of checks (numbers, entities,
proper nouns, state). ``localize(task_id, lang)`` returns a task in the shape the harness bench runs (prompt, tools, verifier, oracle, final,
evidence and state), so every harness, the fake agents and the scoring work on it unchanged.

Every task carries a programmatic verifier and an oracle; a test checks that the oracle passes in both languages under every harness variant with
no extra calls, and that the two versions have the same tool names, argument names, enum values and call structure.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..harnessbench.world import ToolSpec, World
from . import family, shop
from .canon import CANON, LANGS, aliases, fold, label
from .langworld import LangWorld, calculator_spec

SHOP_TOOLS = ["search_products", "get_product", "add_to_cart", "view_cart", "calculator"]
FAMILY_TOOLS = ["family_search", "family_get", "family_total", "family_note", "calculator"]
SEARCH_TOOLS = ("search_products", "family_search")
LANG_ARGS = {"search_products": ["query"], "family_search": ["query"]}
KINDS = ("lookup", "filter", "aggregate", "cart", "recovery", "write", "notool")
PLACEHOLDER = re.compile(r"\{(\w+):([^}]+)\}")


def L(en: Any, es: Any) -> dict[str, Any]:
    return {"en": en, "es": es}


def _spec(task_id: str, kind: str, domain: str, prompt: dict[str, str], oracle: dict[str, list[tuple[str, dict[str, Any]]]], final: dict[str, str], check: dict[str, Any], *,
          evidence: tuple[str, ...] = (), cart: Optional[dict[str, int]] = None, tools: Optional[list[str]] = None) -> dict[str, Any]:
    return {"id": task_id, "kind": kind, "domain": domain, "prompt": prompt, "oracle": oracle, "final": final, "check": check, "evidence": evidence, "cart": cart,
            "tools": tools or (SHOP_TOOLS if domain == "shop" else FAMILY_TOOLS)}


def S(q: str, **kw: Any) -> tuple[str, dict[str, Any]]:
    return ("search_products", {"query": q, **kw})


def FS(collection: str, query: str = "", **kw: Any) -> tuple[str, dict[str, Any]]:
    return ("family_search", {"collection": collection, "query": query, **kw})


def same(*calls: tuple[str, dict[str, Any]]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    return {"en": list(calls), "es": list(calls)}


def pn(pid: str, lang: str) -> str:
    return label("product", pid, lang)


SPECS: list[dict[str, Any]] = [
    # ---------------------------------------------------------------------------------------------------- the shop
    _spec("shop-price-lookup", "lookup", "shop",
          L('How much does the product "{product:P-1001}" cost? Look it up in the shop catalogue and answer with the price in euros.',
            '¿Cuánto cuesta el producto "{product:P-1001}"? Búscalo en el catálogo de la tienda y responde con el precio en euros.'),
          {"en": [S("Nimbus 3 running shoes")], "es": [S("Zapatillas de running Nimbus 3")]},
          L("The Nimbus 3 running shoes cost 89.90 EUR.", "Las zapatillas de running Nimbus 3 cuestan 89.90 EUR."), {"numbers": [89.9]}, evidence=("89.90",)),
    _spec("shop-cheapest-bag", "filter", "shop",
          L("What is the cheapest product in stock in the category \"{category:bags}\"? Give its name and its price.",
            "¿Cuál es el producto más barato con stock de la categoría \"{category:bags}\"? Indica su nombre y su precio."),
          same(S("", category="bags", sort_by="price_asc")),
          L("The cheapest one in stock is the Vuelo waist bag, at 19.90 EUR.", "El más barato con stock es la Riñonera Vuelo, a 19.90 EUR."),
          {"entities": [("product", "P-3005")], "numbers": [19.9]}, evidence=("P-3005", "19.90")),
    _spec("shop-black-audio-budget", "filter", "shop",
          L("I want {color:black} audio products that cost less than 100 euros. Which ones are there? Give their names.",
            "Quiero productos de audio de color {color:black} que cuesten menos de 100 euros. ¿Cuáles hay? Indica sus nombres."),
          {"en": [S("black", category="audio", max_price=100)], "es": [S("negro", category="audio", max_price=100)]},
          L("There is one: the Eco wireless earbuds.", "Hay uno: los Auriculares inalámbricos Eco."),
          {"entities": [("product", "P-4001")], "forbid": [("product", "P-4002"), ("product", "P-4003"), ("product", "P-4005")]}, evidence=("P-4001",)),
    _spec("shop-stock-units", "lookup", "shop",
          L('How many units of "{product:P-2006}" are in stock?', '¿Cuántas unidades de "{product:P-2006}" hay en stock?'),
          {"en": [S("Dune denim jacket")], "es": [S("Chaqueta vaquera Dune")]},
          L("There are 7 units in stock.", "Hay 7 unidades en stock."), {"numbers": [7]}, evidence=("stock 7",)),
    _spec("shop-out-of-stock-answer", "lookup", "shop",
          L('Is "{product:P-2004}" available right now? If it is not, say that it is out of stock.', '¿Está disponible ahora mismo "{product:P-2004}"? Si no lo está, di que está agotado.'),
          {"en": [S("Brezo wool coat")], "es": [S("Abrigo de lana Brezo")]},
          L("No, the Brezo wool coat is out of stock.", "No, el Abrigo de lana Brezo está agotado."),
          {"words": L([["out of stock", "not available", "unavailable", "not in stock", "sold out", "no stock"]], [["agotad", "sin stock", "no está disponible", "no hay stock", "no hay unidades"]])},
          evidence=("stock 0",)),
    _spec("shop-brand", "lookup", "shop",
          L('Which brand makes the product "{product:P-3004}"?', '¿Qué marca fabrica el producto "{product:P-3004}"?'),
          {"en": [S("Cima 45 L trekking backpack"), ("get_product", {"product_id": "P-3004"})], "es": [S("Mochila de trekking Cima 45 L"), ("get_product", {"product_id": "P-3004"})]},
          L("It is made by Terrabruno.", "Lo fabrica Terrabruno."), {"names": ["Terrabruno"]}, evidence=("Terrabruno",)),
    _spec("shop-compare-rating", "aggregate", "shop",
          L('Which has the higher customer rating, "{product:P-4002}" or "{product:P-4001}"? Say which one and by how many rating points.',
            '¿Cuál tiene mejor valoración de los clientes, "{product:P-4002}" o "{product:P-4001}"? Di cuál y por cuántos puntos de valoración.'),
          same(S("Estudio Eco"), ("get_product", {"product_id": "P-4002"}), ("get_product", {"product_id": "P-4001"})),
          L("The Estudio headphones are rated higher, by 0.5 points (4.8 against 4.3).", "Los Auriculares con cancelación de ruido Estudio tienen mejor valoración, por 0.5 puntos (4.8 frente a 4.3)."),
          {"entities": [("product", "P-4002")], "numbers": [0.5]}, evidence=("4.8 / 5", "4.3 / 5")),
    _spec("shop-lamps-total", "aggregate", "shop",
          L('Using the calculator, what do one "{product:P-6001}", one "{product:P-6002}" and one "{product:P-6003}" cost in total?',
            'Usando la calculadora, ¿cuánto cuestan en total un "{product:P-6001}", un "{product:P-6002}" y un "{product:P-6003}"?'),
          same(S("Faro Luna Alba"), ("calculator", {"expression": "39.00 + 25.90 + 84.00"})),
          L("The three lamps cost 148.90 EUR in total.", "Las tres lámparas cuestan 148.90 EUR en total."), {"numbers": [148.9]}, evidence=("148.9",)),
    _spec("shop-flashlight", "lookup", "shop",
          L("Do you sell a rechargeable flashlight? Tell me its price and whether it is in stock.", "¿Venden alguna linterna recargable? Dime su precio y si hay stock."),
          {"en": [S("rechargeable flashlight")], "es": [S("linterna recargable")]},
          L("Yes: the Rayo rechargeable flashlight costs 19.90 EUR, but it is out of stock.", "Sí: la Linterna recargable Rayo cuesta 19.90 EUR, pero está agotada."),
          {"entities": [("product", "P-6005")], "numbers": [19.9],
           "words": L([["out of stock", "not in stock", "unavailable", "not available", "no stock", "sold out"]], [["agotad", "sin stock", "no hay stock", "no está disponible", "no hay unidades"]])},
          evidence=("19.90", "stock 0")),
    _spec("shop-best-rated-kitchen", "filter", "shop",
          L("Which kitchen product has the best customer rating, and how much does it cost?", "¿Qué producto de cocina tiene la mejor valoración de los clientes y cuánto cuesta?"),
          same(S("", category="kitchen", sort_by="rating_desc")),
          L("The Cuchilla chef's knife has the best rating and costs 34.00 EUR.", "El Cuchillo de cocinero Cuchilla tiene la mejor valoración y cuesta 34.00 EUR."),
          {"entities": [("product", "P-5004")], "numbers": [34]}, evidence=("P-5004", "34.00")),
    _spec("shop-cart-boots-slippers", "cart", "shop",
          L('Add 2 units of "{product:P-1003}" and 1 unit of "{product:P-1006}" to my cart, then tell me the cart total.',
            'Añade 2 unidades de "{product:P-1003}" y 1 unidad de "{product:P-1006}" a mi carrito y dime después el total del carrito.'),
          same(S("Alpina Casa"), ("add_to_cart", {"product_id": "P-1003", "quantity": 2}), ("add_to_cart", {"product_id": "P-1006", "quantity": 1})),
          L("Both are in your cart. The cart total is 282.90 EUR.", "Los dos están en tu carrito. El total del carrito es 282.90 EUR."), {"numbers": [282.9]},
          evidence=("282.90",), cart={"P-1003": 2, "P-1006": 1}),
    _spec("shop-fallback-coat", "recovery", "shop",
          L('Add "{product:P-2004}" to my cart. If it is out of stock, add "{product:P-2006}" instead.', 'Añade "{product:P-2004}" a mi carrito. Si está agotado, añade "{product:P-2006}" en su lugar.'),
          same(S("Brezo Dune"), ("add_to_cart", {"product_id": "P-2004", "quantity": 1}), ("add_to_cart", {"product_id": "P-2006", "quantity": 1})),
          L("The Brezo wool coat is out of stock, so I added the Dune denim jacket to your cart.", "El Abrigo de lana Brezo está agotado, así que he añadido la Chaqueta vaquera Dune a tu carrito."),
          {"entities": [("product", "P-2006")]}, evidence=("1 x P-2006",), cart={"P-2006": 1}),
    _spec("shop-gift-budget", "cart", "shop",
          L('I have 100 euros. Add one "{product:P-1004}" and one "{product:P-3001}" to my cart and tell me how much money I have left.',
            'Tengo 100 euros. Añade un "{product:P-1004}" y un "{product:P-3001}" a mi carrito y dime cuánto dinero me queda.'),
          same(S("Brisa Senda"), ("add_to_cart", {"product_id": "P-1004", "quantity": 1}), ("add_to_cart", {"product_id": "P-3001", "quantity": 1}),
               ("calculator", {"expression": "100 - 89.80"})),
          L("Both are in your cart for 89.80 EUR, so you have 10.20 EUR left.", "Los dos están en tu carrito por 89.80 EUR, así que te quedan 10.20 EUR."), {"numbers": [10.2]},
          evidence=("89.80", "10.2"), cart={"P-1004": 1, "P-3001": 1}),
    _spec("shop-explain-warranty", "notool", "shop",
          L("Explain in two sentences what a product warranty is. Do not use any tool.", "Explica en dos frases qué es la garantía de un producto. No uses ninguna herramienta."),
          {"en": [], "es": []},
          L("A warranty is a promise by the seller to repair or replace a product that fails within a set period. It protects the buyer from defects that were not visible when buying it.",
            "Una garantía es la promesa del vendedor de reparar o sustituir un producto que falla dentro de un plazo. Protege al comprador de defectos que no eran visibles al comprarlo."),
          {"words": L([["warranty", "guarantee"]], [["garantía", "garantia"]])}),
    # ---------------------------------------------------------------------------------------------------- the household
    _spec("fam-food-sept", "aggregate", "family",
          L("How much did the household spend on {expense_category:food} in September 2026? Use the household tools.",
            "¿Cuánto gastó el hogar en {expense_category:food} en septiembre de 2026? Usa las herramientas del hogar."),
          same(("family_total", {"month": "2026-09", "category": "food"})),
          L("The household spent 134.30 EUR on groceries in September 2026.", "El hogar gastó 134.30 EUR en alimentación en septiembre de 2026."), {"numbers": [134.3]}, evidence=("134.30",)),
    _spec("fam-biggest-sept", "aggregate", "family",
          L("What was the largest single expense in September 2026, and who paid it?", "¿Cuál fue el gasto individual más grande de septiembre de 2026 y quién lo pagó?"),
          same(FS("expenses", "", month="2026-09")),
          L("The largest expense was Hugo's textbooks, 120.00 EUR, paid by Andrés.", "El gasto más grande fue Libros de texto de Hugo, 120.00 EUR, pagado por Andrés."),
          {"entities": [("concept", "textbooks")], "numbers": [120], "names": ["Andrés"]}, evidence=("G-105",)),
    _spec("fam-dentist", "lookup", "family",
          L("When is Lucía's dentist appointment and where is it?", "¿Cuándo es la cita de Lucía con el dentista y dónde es?"),
          {"en": [FS("appointments", "dentist")], "es": [FS("appointments", "dentista")]},
          L("Lucía's dentist check-up is on 2026-11-03 at 09:30, at Clínica Dental Mirador.", "La revisión del dentista de Lucía es el 2026-11-03 a las 09:30, en Clínica Dental Mirador."),
          {"names": ["Mirador"], "regex": [r"9[:.h]30"]}, evidence=("C-201",)),
    _spec("fam-in-transit", "filter", "family",
          L("Which parcels are in transit right now? Say what each one contains.", "¿Qué envíos están en tránsito ahora mismo? Di qué contiene cada uno."),
          same(FS("shipments")),
          L("Two parcels are in transit: the running shoes (E-301) and the school backpack (E-304).", "Hay dos envíos en tránsito: las zapatillas de running (E-301) y la mochila escolar (E-304)."),
          {"entities": [("parcel", "running_shoes"), ("parcel", "school_backpack")], "forbid": [("parcel", "coffee_maker"), ("parcel", "textbooks")]}, evidence=("E-304",)),
    _spec("fam-delayed", "lookup", "family",
          L("Which parcel is delayed, and which carrier is delivering it?", "¿Qué envío va con retraso y qué transportista lo reparte?"),
          same(FS("shipments")),
          L("The coffee maker is delayed; Nubexpress is delivering it.", "La cafetera va con retraso; la reparte Nubexpress."),
          {"entities": [("parcel", "coffee_maker")], "names": ["Nubexpress"]}, evidence=("delayed",)),
    _spec("fam-expiring", "filter", "family",
          L("Which pantry items expire before 10 November 2026?", "¿Qué productos de la despensa caducan antes del 10 de noviembre de 2026?"),
          same(FS("pantry")),
          L("Three items expire before that date: yogurt, milk and eggs.", "Caducan tres antes de esa fecha: el yogur, la leche y los huevos."),
          {"entities": [("pantry", "yogurt"), ("pantry", "milk"), ("pantry", "eggs")], "forbid": [("pantry", "rice"), ("pantry", "olive_oil"), ("pantry", "lentils")]}, evidence=("D-405",)),
    _spec("fam-utilities-two-months", "aggregate", "family",
          L("How much did the household spend on {expense_category:utilities} in September and October 2026 together?",
            "¿Cuánto gastó el hogar en {expense_category:utilities} en septiembre y octubre de 2026 juntos?"),
          same(("family_total", {"month": "2026-09", "category": "utilities"}), ("family_total", {"month": "2026-10", "category": "utilities"}),
               ("calculator", {"expression": "129.35 + 91.60"})),
          L("Together they came to 220.95 EUR.", "Juntos suman 220.95 EUR."), {"numbers": [220.95]}, evidence=("129.35", "91.60", "220.95")),
    _spec("fam-note-dentist", "write", "family",
          L("Save a note on Lucía's dentist appointment saying to bring the vaccination card.", "Guarda una nota en la cita de Lucía con el dentista que diga que lleve la cartilla de vacunación."),
          {"en": [FS("appointments", "dentist"), ("family_note", {"record_id": "C-201", "note": "Bring the vaccination card"})],
           "es": [FS("appointments", "dentista"), ("family_note", {"record_id": "C-201", "note": "Llevar la cartilla de vacunación"})]},
          L("Done: the note is saved on C-201.", "Hecho: la nota está guardada en C-201."), {"notes": {"C-201": ["vaccin", "vacun"]}}),
    _spec("fam-guitar", "lookup", "family",
          L("At what time is Hugo's guitar lesson?", "¿A qué hora es la clase de guitarra de Hugo?"),
          {"en": [FS("appointments", "guitar")], "es": [FS("appointments", "guitarra")]},
          L("Hugo's guitar lesson is at 18:30.", "La clase de guitarra de Hugo es a las 18:30."), {"regex": [r"18[:.h]30"]}, evidence=("18:30",)),
    _spec("fam-top-payer-oct", "aggregate", "family",
          L("Who paid more in October 2026, Marta or Andrés, and how much did each of them pay in total? Use the calculator for the sums.",
            "¿Quién pagó más en octubre de 2026, Marta o Andrés, y cuánto pagó cada uno en total? Usa la calculadora para las sumas."),
          same(FS("expenses", "", month="2026-10"), ("calculator", {"expression": "58.75 + 60 + 85"}), ("calculator", {"expression": "91.60 + 48 + 33"})),
          L("Marta paid more: 203.75 EUR in total, against 172.60 EUR for Andrés.", "Pagó más Marta: 203.75 EUR en total, frente a 172.60 EUR de Andrés."),
          {"numbers": [203.75, 172.6], "names": ["Marta"]}, evidence=("203.75", "172.6")),
    _spec("fam-cinema", "lookup", "family",
          L("How much did the cinema tickets cost and who bought them?", "¿Cuánto costaron las entradas de cine y quién las compró?"),
          {"en": [FS("expenses", "cinema")], "es": [FS("expenses", "cine")]},
          L("The cinema tickets cost 27.50 EUR and Marta bought them.", "Las entradas de cine costaron 27.50 EUR y las compró Marta."), {"numbers": [27.5], "names": ["Marta"]}, evidence=("G-106",)),
    _spec("fam-arrives-first", "lookup", "family",
          L("Which arrives first, the school backpack or the coffee maker? Give the date.", "¿Qué llega antes, la mochila escolar o la cafetera? Indica la fecha."),
          same(FS("shipments")),
          L("The school backpack arrives first, on 2026-10-31.", "Llega antes la mochila escolar, el 2026-10-31."), {"entities": [("parcel", "school_backpack")], "regex": [r"\b31\b"]}, evidence=("E-304",)),
    _spec("fam-explain-budget", "notool", "family",
          L("Explain in a short paragraph why it is useful for a family to keep a monthly budget. Do not use any tool.",
            "Explica en un párrafo corto por qué es útil para una familia llevar un presupuesto mensual. No uses ninguna herramienta."),
          {"en": [], "es": []},
          L("A monthly budget shows a family where its money goes before it is spent. That makes it easier to save, to plan big purchases and to avoid surprises at the end of the month.",
            "Un presupuesto mensual muestra a una familia adónde va su dinero antes de gastarlo. Así es más fácil ahorrar, planificar las compras grandes y evitar sorpresas a final de mes."),
          {"words": L([["budget"]], [["presupuesto"]])}),
]
SPECS_BY_ID = {s["id"]: s for s in SPECS}
assert len(SPECS_BY_ID) == len(SPECS)


# ------------------------------------------------------------------------------------------------- the checks
def resolve(text: str, lang: str) -> str:
    """Replace ``{kind:key}`` placeholders with the canonical label of the language."""
    return PLACEHOLDER.sub(lambda m: label(m.group(1), m.group(2), lang), text)


def _has(folded: str, phrase: str) -> bool:
    """Is the phrase in the (folded) text as whole words, each of them possibly in the plural?"""
    body = r"\s+".join(re.escape(w) + "(?:es|s)?" for w in fold(phrase).split())
    return re.search(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", folded) is not None


def mentions(text: str, kind: str, key: str, lang: str) -> bool:
    """Does the text name the entity? Its localised label, one of its aliases, or (a product) its model name."""
    folded = fold(text)
    names = aliases(kind, key, lang)
    if kind == "product":
        names = names + [shop.BY_ID[key]["model"]]
    return any(_has(folded, n) for n in names)


def make_verify(check: dict[str, Any], lang: str, cart: Optional[dict[str, int]]) -> Callable[[str, World], tuple[bool, str]]:
    from ..harnessbench.tasks import has_number

    def verify(final: str, world: World) -> tuple[bool, str]:
        folded = fold(final)
        w = world if isinstance(world, LangWorld) else None
        if cart is not None and (w is None or w.cart != cart):
            return False, f"the cart does not hold {cart}"
        for rec, needles in (check.get("notes") or {}).items():
            note = fold((w.notes.get(rec, "") if w else ""))
            if not note or not any(fold(n) in note for n in needles):
                return False, f"the note on {rec} is missing or does not say the right thing"
        if check.get("notes") and not check.get("numbers") and not check.get("entities") and not check.get("words"):
            return True, ""
        miss = [f"{v:g}" for v in check.get("numbers", []) if not has_number(final, v, 0.005)]
        if miss:
            return False, f"the answer lacks {', '.join(miss)}"
        gone = [key for kind, key in check.get("entities", []) if not mentions(final, kind, key, lang)]
        if gone:
            return False, f"the answer does not name {', '.join(label(k, x, lang) for k, x in check.get('entities', []) if x in gone)}"
        extra = [key for kind, key in check.get("forbid", []) if mentions(final, kind, key, lang)]
        if extra:
            return False, f"the answer names something it should not: {', '.join(extra)}"
        lost = [n for n in check.get("names", []) if not _has(folded, n)]
        if lost:
            return False, f"the answer lacks {', '.join(lost)}"
        for group in (check.get("words") or {}).get(lang, []):
            if not any(fold(g) in folded for g in group):
                return False, f"the answer lacks one of: {', '.join(group)}"
        for pattern in check.get("regex", []):
            if not re.search(pattern, folded):
                return False, f"the answer does not match {pattern}"
        return True, ""
    return verify


def _state(cart: Optional[dict[str, int]], check: dict[str, Any], lang: str) -> Optional[Callable[[World], bool]]:
    if cart is None and not check.get("notes"):
        return None

    def state(world: World) -> bool:
        w = world if isinstance(world, LangWorld) else None
        if w is None:
            return False
        if cart is not None and w.cart != cart:
            return False
        for rec, needles in (check.get("notes") or {}).items():
            note = fold(w.notes.get(rec, ""))
            if not note or not any(fold(n) in note for n in needles):
                return False
        return True
    return state


_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def specs_for(domain: str, tools: list[str], lang: str) -> dict[str, ToolSpec]:
    table = {**(shop.specs(lang) if domain == "shop" else family.specs(lang)), "calculator": calculator_spec(lang)}
    return {n: table[n] for n in tools}


def localize(task_id: str, lang: str) -> dict[str, Any]:
    """The task ``task_id`` in ``lang``, in the shape the harness bench runs."""
    key = (task_id, lang)
    if key not in _CACHE:
        s = SPECS_BY_ID[task_id]
        _CACHE[key] = {
            "id": s["id"], "kind": s["kind"], "domain": s["domain"], "set": "i18n", "lang": lang, "prompt": resolve(s["prompt"][lang], lang), "tools": list(s["tools"]),
            "specs": specs_for(s["domain"], s["tools"], lang), "verify": make_verify(s["check"], lang, s["cart"]), "oracle": [(n, dict(a)) for n, a in s["oracle"][lang]],
            "final": s["final"][lang], "evidence": list(s["evidence"]), "state": _state(s["cart"], s["check"], lang), "world": {}, "external": False,
            "make_world": (lambda lg=lang: LangWorld(lg)), "search_tools": list(SEARCH_TOOLS), "lang_args": {k: list(v) for k, v in LANG_ARGS.items()},
        }
    return _CACHE[key]


I18N_TASK_IDS = [s["id"] for s in SPECS]


def select(only: Any = (), kinds: Any = ()) -> list[str]:
    only, kinds = set(only or ()), set(kinds or ())
    return [s["id"] for s in SPECS if (not only or s["id"] in only) and (not kinds or s["kind"] in kinds)]


def public(task_id: str) -> dict[str, Any]:
    s = SPECS_BY_ID[task_id]
    return {"id": s["id"], "kind": s["kind"], "domain": s["domain"], "set": "i18n", "langs": list(LANGS), "tools": list(s["tools"]), "external": False,
            "prompt": {lang: resolve(s["prompt"][lang], lang) for lang in LANGS}, "oracle_calls": len(s["oracle"]["en"])}


__all__ = ["I18N_TASK_IDS", "KINDS", "LANGS", "SPECS", "SPECS_BY_ID", "localize", "mentions", "public", "resolve", "select"]
