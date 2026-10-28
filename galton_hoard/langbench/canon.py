"""The canonical entity map of the language bench and the three ways a string is localised.

A task exists in two languages (Spanish and English) with the same ids. What differs between the two is decided by the type of each string:

* **Preserve**: never translated, so the two versions can be compared call by call. Ids (``P-1003``, ``C-201``), tool names, argument names, enum
  values (``category=shoes``, ``collection=expenses``, ``month=2026-09``), brands, places, people's names, prices and dates.
* **Translate-Free**: prose with no fixed form, written freshly in each language. Task instructions, tool descriptions, argument descriptions, error
  messages, product descriptions.
* **Translate-Canonical**: an entity that appears in the instructions, in the tool results (the data) and in the checks of the same task. It is
  translated once, here, and every one of those places reads the same entry: the product named in the Spanish prompt is the one the Spanish tool
  returns and the one the Spanish verifier looks for. Product names, categories, colours, expense categories, record titles, pantry items.

This module is the single place for the third type. Nothing else in the bench writes a localised entity by hand.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

LANGS = ("en", "es")
BASE_LANG = "en"
LANG_NAMES = {"en": ("English", "inglés"), "es": ("Spanish", "castellano")}

PRESERVE, TRANSLATE_FREE, TRANSLATE_CANONICAL = "preserve", "translate_free", "translate_canonical"
TYPOLOGY: dict[str, dict[str, str]] = {
    PRESERVE: {"en": "ids, tool and argument names, enum values, brands, places, people, prices, dates: identical in both versions",
               "es": "ids, nombres de herramientas y argumentos, valores de enumerados, marcas, lugares, personas, precios y fechas: idénticos en las dos versiones"},
    TRANSLATE_FREE: {"en": "instructions, tool and argument descriptions, messages: written afresh in each language",
                     "es": "instrucciones, descripciones de herramientas y argumentos, mensajes: se redactan de nuevo en cada idioma"},
    TRANSLATE_CANONICAL: {"en": "entities shared by prompts, tool data and checks: translated once in the canonical map and read from there",
                          "es": "entidades que comparten instrucciones, datos de las herramientas y comprobaciones: se traducen una vez en el mapa canónico y se leen de él"},
}


def _e(en: str, es: str, *, en_alias: Iterable[str] = (), es_alias: Iterable[str] = ()) -> dict[str, object]:
    return {"en": en, "es": es, "alias": {"en": list(en_alias), "es": list(es_alias)}}


#: kind -> key -> entry. The keys are Preserve strings; the labels are Translate-Canonical.
CANON: dict[str, dict[str, dict[str, object]]] = {
    "category": {
        "shoes": _e("shoes", "calzado", es_alias=("zapatillas",)),
        "jackets": _e("jackets", "chaquetas", es_alias=("abrigos",)),
        "bags": _e("bags and backpacks", "mochilas y bolsos", en_alias=("backpacks", "bags"), es_alias=("mochilas", "bolsos")),
        "audio": _e("audio", "audio", en_alias=("headphones",), es_alias=("auriculares",)),
        "kitchen": _e("kitchen", "cocina"),
        "lighting": _e("lighting", "iluminación", en_alias=("lamps",), es_alias=("lámparas",)),
    },
    "color": {
        "black": _e("black", "negro"), "white": _e("white", "blanco"), "brown": _e("brown", "marrón"), "blue": _e("blue", "azul"),
        "green": _e("green", "verde"), "grey": _e("grey", "gris", en_alias=("gray",)), "yellow": _e("yellow", "amarillo"), "navy": _e("navy", "azul marino"),
        "red": _e("red", "rojo"), "sand": _e("sand", "arena"), "orange": _e("orange", "naranja"), "steel": _e("steel", "acero"), "wood": _e("wood", "madera"),
    },
    "product": {
        "P-1001": _e("Nimbus 3 running shoes", "Zapatillas de running Nimbus 3"),
        "P-1002": _e("Corsa city sneakers", "Zapatillas urbanas Corsa"),
        "P-1003": _e("Alpina leather hiking boots", "Botas de montaña de cuero Alpina"),
        "P-1004": _e("Brisa beach sandals", "Sandalias de playa Brisa"),
        "P-1005": _e("Nimbus Trail trail running shoes", "Zapatillas de trail running Nimbus Trail"),
        "P-1006": _e("Casa wool slippers", "Pantuflas de lana Casa"),
        "P-2001": _e("Nordic down jacket", "Plumífero Nordic"),
        "P-2002": _e("Lluvia waterproof raincoat", "Impermeable Lluvia"),
        "P-2003": _e("Ronda softshell jacket", "Chaqueta softshell Ronda"),
        "P-2004": _e("Brezo wool coat", "Abrigo de lana Brezo"),
        "P-2005": _e("Aire windbreaker", "Cortavientos Aire"),
        "P-2006": _e("Dune denim jacket", "Chaqueta vaquera Dune"),
        "P-3001": _e("Senda 25 L hiking backpack", "Mochila de senderismo Senda 25 L"),
        "P-3002": _e("Urbe laptop backpack", "Mochila para portátil Urbe"),
        "P-3003": _e("Cole school backpack", "Mochila escolar Cole"),
        "P-3004": _e("Cima 45 L trekking backpack", "Mochila de trekking Cima 45 L"),
        "P-3005": _e("Vuelo waist bag", "Riñonera Vuelo"),
        "P-3006": _e("Ruta cycling backpack", "Mochila de ciclismo Ruta"),
        "P-4001": _e("Eco wireless earbuds", "Auriculares inalámbricos Eco"),
        "P-4002": _e("Estudio noise-cancelling headphones", "Auriculares con cancelación de ruido Estudio"),
        "P-4003": _e("Pulso wired earphones", "Auriculares con cable Pulso"),
        "P-4004": _e("Ritmo sports earbuds", "Auriculares deportivos Ritmo"),
        "P-4005": _e("Sala bluetooth speaker", "Altavoz bluetooth Sala"),
        "P-4006": _e("Cinta over-ear headband headphones", "Auriculares de diadema Cinta"),
        "P-5001": _e("Hierro cast-iron skillet", "Sartén de hierro fundido Hierro"),
        "P-5002": _e("Brasa pressure cooker", "Olla a presión Brasa"),
        "P-5003": _e("Molino manual coffee grinder", "Molinillo de café manual Molino"),
        "P-5004": _e("Cuchilla chef's knife", "Cuchillo de cocinero Cuchilla"),
        "P-5005": _e("Hervor electric kettle", "Hervidor eléctrico Hervor"),
        "P-5006": _e("Pan bamboo cutting board", "Tabla de cortar de bambú Pan"),
        "P-6001": _e("Faro desk lamp", "Lámpara de escritorio Faro"),
        "P-6002": _e("Luna bedside lamp", "Lámpara de mesita Luna"),
        "P-6003": _e("Alba LED floor lamp", "Lámpara de pie LED Alba"),
        "P-6004": _e("Vela string lights", "Guirnalda de luces Vela"),
        "P-6005": _e("Rayo rechargeable flashlight", "Linterna recargable Rayo"),
        "P-6006": _e("Bruma smart bulb kit", "Kit de bombillas inteligentes Bruma"),
    },
    # -- the family data -----------------------------------------------------------------------------------------------
    "expense_category": {
        "food": _e("groceries", "alimentación", en_alias=("food",)), "utilities": _e("utilities", "suministros"), "health": _e("health", "salud"),
        "transport": _e("transport", "transporte"), "leisure": _e("leisure", "ocio"), "education": _e("education", "educación"),
    },
    "concept": {
        "weekly_shop": _e("Weekly shop at Olivar", "Compra semanal en Olivar"),
        "electricity": _e("Electricity bill", "Factura de la luz"),
        "water": _e("Water bill", "Factura del agua"),
        "medicines": _e("Lucía's medicines", "Medicinas de Lucía"),
        "petrol": _e("Petrol", "Gasolina"),
        "textbooks": _e("Hugo's textbooks", "Libros de texto de Hugo", en_alias=("textbooks",), es_alias=("libros de texto",)),
        "cinema": _e("Cinema tickets", "Entradas de cine", en_alias=("cinema",), es_alias=("cine",)),
        "dentist_fee": _e("Dentist check-up fee", "Revisión del dentista"),
        "birthday": _e("Hugo's birthday dinner", "Cena de cumpleaños de Hugo", en_alias=("birthday",), es_alias=("cumpleaños",)),
        "english_course": _e("Lucía's English course fee", "Matrícula del curso de inglés de Lucía", en_alias=("english course",), es_alias=("curso de inglés",)),
    },
    "appointment": {
        "dentist": _e("Dentist check-up", "Revisión del dentista", en_alias=("dentist",), es_alias=("dentista",)),
        "parents_meeting": _e("Parents' meeting at school", "Reunión de padres en el colegio", en_alias=("parents' meeting",), es_alias=("reunión de padres",)),
        "physio": _e("Physiotherapist appointment", "Cita con el fisioterapeuta", en_alias=("physiotherapist",), es_alias=("fisioterapeuta",)),
        "guitar": _e("Guitar lesson", "Clase de guitarra", en_alias=("guitar",), es_alias=("guitarra",)),
        "mot": _e("Car inspection", "Revisión del coche", en_alias=("inspection",), es_alias=("revisión del coche", "itv")),
        "english_exam": _e("English exam", "Examen de inglés", en_alias=("exam",), es_alias=("examen",)),
    },
    "parcel": {
        "running_shoes": _e("running shoes", "zapatillas de running"),
        "textbooks": _e("textbooks", "libros de texto"),
        "coffee_maker": _e("coffee maker", "cafetera"),
        "school_backpack": _e("school backpack", "mochila escolar"),
    },
    "pantry": {
        "milk": _e("milk", "leche"), "rice": _e("rice", "arroz"), "eggs": _e("eggs", "huevos"), "olive_oil": _e("olive oil", "aceite de oliva"),
        "yogurt": _e("yogurt", "yogur", en_alias=("yoghurt",)), "lentils": _e("lentils", "lentejas"),
    },
    "unit": {"l": _e("l", "l"), "kg": _e("kg", "kg"), "u": _e("pcs", "uds")},
    "term": {                       # words the tools print around the data
        "stock": _e("stock", "stock"), "rating": _e("rating", "valoración"), "expires": _e("expires", "caduca"), "eta": _e("ETA", "llegada"),
    },
}

#: Preserve strings that appear in prose: brands, places, people. They are masked when the language of a reply is judged.
BRANDS = ("Nordvik", "Solana", "Terrabruno", "Aurelia", "Cobalto", "Marea")
PLACES = ("Olivar", "Clínica Dental Mirador", "Centro Fisio Norte", "ITV Poniente", "Colegio San Remo", "Academia Sol y Cuerda", "Rutapost", "Nubexpress")
PEOPLE = ("Marta", "Andrés", "Lucía", "Hugo")


def fold(text: str) -> str:
    """Lower case, no accents: the form in which strings are compared."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))


def label(kind: str, key: str, lang: str) -> str:
    """The localised name of an entity: the one place a Translate-Canonical string is read."""
    return str(CANON[kind][key][lang])


def aliases(kind: str, key: str, lang: str) -> list[str]:
    entry = CANON[kind][key]
    return [str(entry[lang]), *entry["alias"][lang]]                      # type: ignore[index]


def kinds_and_keys() -> list[tuple[str, str]]:
    return [(kind, key) for kind, table in CANON.items() for key in table]


NAMED_KINDS = ("product", "concept", "appointment")         # titles that are quoted as they are in the data; common nouns (colours, pantry items...) are prose


def every_label() -> list[str]:
    """Every localised title and every Preserve proper noun: what a reply may quote without it counting as prose."""
    out: list[str] = list(BRANDS) + list(PLACES) + list(PEOPLE)
    for kind in NAMED_KINDS:
        for entry in CANON[kind].values():
            for lang in LANGS:
                out.append(str(entry[lang]))
    return sorted({x for x in out if x}, key=len, reverse=True)


_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))
