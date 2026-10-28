"""The shop of the language bench: a 36-product catalogue and the tools that search and fill a cart, in Spanish and English.

The attributes (id, brand, price, stock, rating, category and colour keys) are Preserve. The names, category and colour labels come from the
canonical map; the descriptions and every message the tools print are Translate-Free. Search is BM25 over the localised text, so a query in the
other language than the catalogue finds nothing.
"""

from __future__ import annotations

from typing import Any

from ..harnessbench.world import ToolSpec, _ToolFail
from .bm25 import Bm25
from .canon import CANON, label

# id, category, brand, colour, price (EUR), stock, rating, model
PRODUCTS: list[dict[str, Any]] = [dict(zip(("id", "category", "brand", "color", "price", "stock", "rating", "model"), row)) for row in (
    ("P-1001", "shoes", "Solana", "black", 89.90, 14, 4.6, "Nimbus 3"), ("P-1002", "shoes", "Solana", "white", 64.50, 0, 4.2, "Corsa"),
    ("P-1003", "shoes", "Terrabruno", "brown", 129.00, 6, 4.8, "Alpina"), ("P-1004", "shoes", "Marea", "blue", 39.90, 22, 3.9, "Brisa"),
    ("P-1005", "shoes", "Solana", "green", 104.90, 9, 4.4, "Nimbus Trail"), ("P-1006", "shoes", "Aurelia", "grey", 24.90, 31, 4.1, "Casa"),
    ("P-2001", "jackets", "Nordvik", "black", 149.00, 8, 4.7, "Nordic"), ("P-2002", "jackets", "Marea", "yellow", 59.90, 18, 4.3, "Lluvia"),
    ("P-2003", "jackets", "Terrabruno", "navy", 79.00, 11, 4.0, "Ronda"), ("P-2004", "jackets", "Aurelia", "green", 119.00, 0, 4.5, "Brezo"),
    ("P-2005", "jackets", "Marea", "red", 34.90, 40, 3.8, "Aire"), ("P-2006", "jackets", "Cobalto", "sand", 89.50, 7, 4.4, "Dune"),
    ("P-3001", "bags", "Terrabruno", "green", 49.90, 20, 4.5, "Senda"), ("P-3002", "bags", "Cobalto", "black", 69.00, 15, 4.6, "Urbe"),
    ("P-3003", "bags", "Marea", "red", 29.90, 25, 4.0, "Cole"), ("P-3004", "bags", "Terrabruno", "orange", 139.00, 4, 4.7, "Cima"),
    ("P-3005", "bags", "Cobalto", "black", 19.90, 33, 3.9, "Vuelo"), ("P-3006", "bags", "Solana", "grey", 54.90, 10, 4.2, "Ruta"),
    ("P-4001", "audio", "Cobalto", "black", 59.00, 30, 4.3, "Eco"), ("P-4002", "audio", "Cobalto", "black", 179.00, 5, 4.8, "Estudio"),
    ("P-4003", "audio", "Marea", "red", 29.90, 50, 3.7, "Pulso"), ("P-4004", "audio", "Solana", "white", 99.00, 12, 4.4, "Ritmo"),
    ("P-4005", "audio", "Cobalto", "grey", 79.90, 0, 4.1, "Sala"), ("P-4006", "audio", "Aurelia", "blue", 49.00, 17, 4.0, "Cinta"),
    ("P-5001", "kitchen", "Aurelia", "black", 44.90, 14, 4.6, "Hierro"), ("P-5002", "kitchen", "Nordvik", "steel", 79.00, 9, 4.5, "Brasa"),
    ("P-5003", "kitchen", "Aurelia", "white", 24.50, 27, 4.2, "Molino"), ("P-5004", "kitchen", "Nordvik", "steel", 34.00, 19, 4.7, "Cuchilla"),
    ("P-5005", "kitchen", "Cobalto", "red", 29.00, 16, 4.0, "Hervor"), ("P-5006", "kitchen", "Aurelia", "wood", 18.90, 35, 4.3, "Pan"),
    ("P-6001", "lighting", "Nordvik", "black", 39.00, 13, 4.2, "Faro"), ("P-6002", "lighting", "Aurelia", "white", 25.90, 21, 4.0, "Luna"),
    ("P-6003", "lighting", "Nordvik", "black", 84.00, 6, 4.5, "Alba"), ("P-6004", "lighting", "Marea", "yellow", 12.90, 60, 3.8, "Vela"),
    ("P-6005", "lighting", "Solana", "grey", 19.90, 0, 4.1, "Rayo"), ("P-6006", "lighting", "Cobalto", "white", 49.90, 8, 4.4, "Bruma"),
)]
BY_ID = {p["id"]: p for p in PRODUCTS}
CATEGORIES = tuple(CANON["category"])
SORTS = ("relevance", "price_asc", "price_desc", "rating_desc")
LIMIT = 8

DESCRIPTION = {         # Translate-Free: one sentence per category and language
    "shoes": {"en": "Cushioned insole and a grippy rubber sole.", "es": "Plantilla acolchada y suela de goma con buen agarre."},
    "jackets": {"en": "Lined, with two zipped pockets.", "es": "Con forro y dos bolsillos con cremallera."},
    "bags": {"en": "Padded straps and a water-resistant fabric.", "es": "Correas acolchadas y tejido resistente al agua."},
    "audio": {"en": "Bluetooth 5 with a carrying case.", "es": "Bluetooth 5 con funda de transporte."},
    "kitchen": {"en": "Dishwasher safe and built to last.", "es": "Apto para lavavajillas y de larga duración."},
    "lighting": {"en": "Low energy use and a warm light.", "es": "Bajo consumo y luz cálida."},
}

MSG = {          # Translate-Free: what the tools print
    "en": {"none": "No products found for '{q}'.", "list_head": "{n} product(s):", "cart_empty": "The cart is empty.", "added": "Added {n} x {id} to the cart. Cart total: {total} EUR",
           "oos": "{id} is out of stock.", "few": "only {n} unit(s) of {id} in stock.", "no_product": "no product '{id}'.", "total": "Total: {total} EUR", "bad_qty": "the quantity must be 1 or more.",
           "name": "Name", "brand": "Brand", "category": "Category", "color": "Colour", "price": "Price", "stock": "Stock", "rating": "Rating", "about": "Description"},
    "es": {"none": "No se encontraron productos para '{q}'.", "list_head": "{n} producto(s):", "cart_empty": "El carrito está vacío.", "added": "Añadidos {n} x {id} al carrito. Total del carrito: {total} EUR",
           "oos": "{id} está agotado.", "few": "solo quedan {n} unidad(es) de {id}.", "no_product": "no existe el producto '{id}'.", "total": "Total: {total} EUR", "bad_qty": "la cantidad debe ser 1 o más.",
           "name": "Nombre", "brand": "Marca", "category": "Categoría", "color": "Color", "price": "Precio", "stock": "Stock", "rating": "Valoración", "about": "Descripción"},
}

TOOLS_TEXT: dict[str, dict[str, Any]] = {       # Translate-Free: descriptions of the tools and of their arguments (the names and enums are Preserve)
    "search_products": {
        "en": ("Search the shop catalogue by words. Returns up to 8 matching products with id, name, price and stock.",
               {"query": "Words to look for in the product names, for example a kind of product or a colour. Use the language of the catalogue.", "category": "Only this category.",
                "max_price": "Only products that cost at most this many euros.", "sort_by": "Order of the results."}),
        "es": ("Busca en el catálogo de la tienda por palabras. Devuelve hasta 8 productos con id, nombre, precio y stock.",
               {"query": "Palabras que buscar en los nombres de los productos, por ejemplo un tipo de producto o un color. Usa el idioma del catálogo.", "category": "Solo esta categoría.",
                "max_price": "Solo productos que cuesten como mucho esta cantidad de euros.", "sort_by": "Orden de los resultados."})},
    "get_product": {
        "en": ("Read the full details of one product: name, brand, category, colour, price, stock, rating and description.", {"product_id": "Product id, for example P-1001."}),
        "es": ("Lee todos los datos de un producto: nombre, marca, categoría, color, precio, stock, valoración y descripción.", {"product_id": "Id del producto, por ejemplo P-1001."})},
    "add_to_cart": {
        "en": ("Add units of a product to the cart. Fails when the product is out of stock or there are not enough units.", {"product_id": "Product id.", "quantity": "Number of units."}),
        "es": ("Añade unidades de un producto al carrito. Falla si el producto está agotado o no hay unidades suficientes.", {"product_id": "Id del producto.", "quantity": "Número de unidades."})},
    "view_cart": {"en": ("Show the cart: lines and total.", {}), "es": ("Muestra el carrito: líneas y total.", {})},
}


def _text_of(p: dict[str, Any], lang: str) -> str:
    cat = CANON["category"][p["category"]]
    return " ".join([label("product", p["id"], lang), str(cat[lang]), *cat["alias"][lang], label("color", p["color"], lang), p["brand"], p["model"]])      # type: ignore[arg-type,list-item]


_INDEX: dict[str, Bm25] = {}


def index(lang: str) -> Bm25:
    if lang not in _INDEX:
        _INDEX[lang] = Bm25([_text_of(p, lang) for p in PRODUCTS])
    return _INDEX[lang]


def specs(lang: str) -> dict[str, ToolSpec]:
    def spec(name: str, props: dict[str, dict[str, Any]], required: list[str]) -> ToolSpec:
        desc, pdesc = TOOLS_TEXT[name][lang]
        return ToolSpec(name, desc, {"type": "object", "properties": {k: {**v, "description": pdesc[k]} for k, v in props.items()}, "required": required})
    return {
        "search_products": spec("search_products", {"query": {"type": "string"}, "category": {"type": "string", "enum": list(CATEGORIES)}, "max_price": {"type": "number"},
                                                    "sort_by": {"type": "string", "enum": list(SORTS)}}, ["query"]),
        "get_product": spec("get_product", {"product_id": {"type": "string"}}, ["product_id"]),
        "add_to_cart": spec("add_to_cart", {"product_id": {"type": "string"}, "quantity": {"type": "number"}}, ["product_id", "quantity"]),
        "view_cart": spec("view_cart", {}, []),
    }


def money(x: float) -> str:
    return f"{x:.2f}"


class ShopTools:
    """The shop tools; mixed into ``LangWorld`` (which owns ``lang``, ``cart`` and the ``_empty`` flag)."""

    lang: str
    cart: dict[str, int]
    _empty: bool

    # -- the shop ------------------------------------------------------------------------------------------------------
    def _line(self, p: dict[str, Any]) -> str:
        return f"{p['id']} | {label('product', p['id'], self.lang)} | {money(p['price'])} EUR | {label('term', 'stock', self.lang)} {p['stock']}"

    def _t_search_products(self, query: str, category: str = "", max_price: float = 0, sort_by: str = "relevance") -> str:
        m = MSG[self.lang]
        order = index(self.lang).rank(query) if str(query).strip() else list(range(len(PRODUCTS)))
        rows = [PRODUCTS[i] for i in order]
        if category:
            rows = [p for p in rows if p["category"] == category]
        if max_price:
            rows = [p for p in rows if p["price"] <= float(max_price)]
        if sort_by == "price_asc":
            rows.sort(key=lambda p: p["price"])
        elif sort_by == "price_desc":
            rows.sort(key=lambda p: -p["price"])
        elif sort_by == "rating_desc":
            rows.sort(key=lambda p: -p["rating"])
        rows = rows[:LIMIT]
        if not rows:
            self._empty = True
            return m["none"].format(q=query)
        return m["list_head"].format(n=len(rows)) + "\n" + "\n".join(self._line(p) for p in rows)

    def _product(self, product_id: str) -> dict[str, Any]:
        p = BY_ID.get(str(product_id).strip().upper())
        if p is None:
            raise _ToolFail(MSG[self.lang]["no_product"].format(id=product_id))
        return p

    def _t_get_product(self, product_id: str) -> str:
        m, p = MSG[self.lang], self._product(product_id)
        return "\n".join([p["id"], f"{m['name']}: {label('product', p['id'], self.lang)}", f"{m['brand']}: {p['brand']}", f"{m['category']}: {label('category', p['category'], self.lang)}",
                          f"{m['color']}: {label('color', p['color'], self.lang)}", f"{m['price']}: {money(p['price'])} EUR", f"{m['stock']}: {p['stock']}",
                          f"{m['rating']}: {p['rating']} / 5", f"{m['about']}: {DESCRIPTION[p['category']][self.lang]}"])

    def _t_add_to_cart(self, product_id: str, quantity: float) -> str:
        m, p = MSG[self.lang], self._product(product_id)
        n = int(quantity)
        if n < 1:
            raise _ToolFail(m["bad_qty"])
        if p["stock"] == 0:
            raise _ToolFail(m["oos"].format(id=p["id"]))
        if self.cart.get(p["id"], 0) + n > p["stock"]:
            raise _ToolFail(m["few"].format(n=p["stock"], id=p["id"]))
        self.cart[p["id"]] = self.cart.get(p["id"], 0) + n
        return m["added"].format(n=n, id=p["id"], total=money(self.cart_total()))

    def cart_total(self) -> float:
        return round(sum(BY_ID[i]["price"] * n for i, n in self.cart.items()), 2)

    def _t_view_cart(self) -> str:
        m = MSG[self.lang]
        if not self.cart:
            return m["cart_empty"]
        lines = [f"{i} x{n} = {money(BY_ID[i]['price'] * n)} EUR" for i, n in self.cart.items()]
        return "\n".join([*lines, m["total"].format(total=money(self.cart_total()))])


__all__ = ["BY_ID", "CATEGORIES", "MSG", "PRODUCTS", "SORTS", "ShopTools", "index", "specs"]
