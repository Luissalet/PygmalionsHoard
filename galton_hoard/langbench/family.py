"""The family data of the language bench: expenses, appointments, parcels and the pantry of an invented household, written in Spanish.

The records are native Spanish data. The English version is the same records read through the canonical map, so an English search finds
nothing in the Spanish run and the other way round. Ids, dates, amounts, people, places, carriers and statuses are Preserve; the concepts, titles,
parcel contents and pantry items come from the canonical map; the messages and descriptions of the tools are Translate-Free.
"""

from __future__ import annotations

from typing import Any

from ..harnessbench.world import ToolSpec, _ToolFail
from .bm25 import Bm25
from .canon import CANON, label

COLLECTIONS = ("expenses", "appointments", "shipments", "pantry")
EXPENSE_CATEGORIES = tuple(CANON["expense_category"])
STATUSES = ("in_transit", "delivered", "delayed")
LIMIT = 15

EXPENSES = [dict(zip(("id", "date", "category", "concept", "payer", "amount"), r)) for r in (
    ("G-101", "2026-09-03", "food", "weekly_shop", "Marta", 62.40), ("G-102", "2026-09-05", "utilities", "electricity", "Andrés", 88.15),
    ("G-103", "2026-09-09", "health", "medicines", "Marta", 35.00), ("G-104", "2026-09-12", "transport", "petrol", "Andrés", 54.30),
    ("G-105", "2026-09-15", "education", "textbooks", "Andrés", 120.00), ("G-106", "2026-09-18", "leisure", "cinema", "Marta", 27.50),
    ("G-107", "2026-09-21", "food", "weekly_shop", "Andrés", 71.90), ("G-108", "2026-09-27", "utilities", "water", "Marta", 41.20),
    ("G-109", "2026-10-02", "food", "weekly_shop", "Marta", 58.75), ("G-110", "2026-10-06", "utilities", "electricity", "Andrés", 91.60),
    ("G-111", "2026-10-10", "health", "dentist_fee", "Andrés", 48.00), ("G-112", "2026-10-14", "transport", "petrol", "Marta", 60.00),
    ("G-113", "2026-10-17", "leisure", "birthday", "Andrés", 33.00), ("G-114", "2026-10-20", "education", "english_course", "Marta", 85.00))]
APPOINTMENTS = [dict(zip(("id", "date", "time", "person", "title", "place"), r)) for r in (
    ("C-201", "2026-11-03", "09:30", "Lucía", "dentist", "Clínica Dental Mirador"), ("C-202", "2026-11-05", "17:00", "Andrés", "parents_meeting", "Colegio San Remo"),
    ("C-203", "2026-11-07", "11:15", "Marta", "physio", "Centro Fisio Norte"), ("C-204", "2026-11-12", "18:30", "Hugo", "guitar", "Academia Sol y Cuerda"),
    ("C-205", "2026-11-14", "10:00", "Andrés", "mot", "ITV Poniente"), ("C-206", "2026-11-20", "16:00", "Lucía", "english_exam", "Colegio San Remo"))]
SHIPMENTS = [dict(zip(("id", "item", "carrier", "status", "date"), r)) for r in (
    ("E-301", "running_shoes", "Rutapost", "in_transit", "2026-10-30"), ("E-302", "textbooks", "Rutapost", "delivered", "2026-10-22"),
    ("E-303", "coffee_maker", "Nubexpress", "delayed", "2026-11-04"), ("E-304", "school_backpack", "Nubexpress", "in_transit", "2026-10-31"))]
PANTRY = [dict(zip(("id", "item", "qty", "unit", "date"), r)) for r in (
    ("D-401", "milk", 6, "l", "2026-11-02"), ("D-402", "rice", 2, "kg", "2027-03-01"), ("D-403", "eggs", 12, "u", "2026-11-05"),
    ("D-404", "olive_oil", 1, "l", "2027-01-15"), ("D-405", "yogurt", 8, "u", "2026-10-29"), ("D-406", "lentils", 1, "kg", "2027-06-30"))]
DATA = {"expenses": EXPENSES, "appointments": APPOINTMENTS, "shipments": SHIPMENTS, "pantry": PANTRY}
BY_ID = {r["id"]: (c, r) for c, rows in DATA.items() for r in rows}

MSG = {
    "en": {"none": "No records match '{q}' in {c}.", "none_month": "No records in {c} for {m}.", "head": "{n} record(s) in {c}:", "no_record": "no record '{id}'.", "noted": "Note saved on {id}.",
           "note": "Note", "total_all": "Total {m}: {t} EUR ({n} expense(s))", "total_cat": "Total {cat} {m}: {t} EUR ({n} expense(s))", "no_expenses": "No expenses for that selection: total 0.00 EUR.",
           "bad_month": "the month must be written YYYY-MM, for example 2026-09.", "empty_note": "the note is empty.", "person": "Person", "place": "Place", "carrier": "Carrier",
           "status": "Status", "date": "Date", "amount": "Amount", "payer": "Paid by", "category": "Category", "concept": "Concept", "item": "Item", "qty": "Quantity"},
    "es": {"none": "Ningún registro coincide con '{q}' en {c}.", "none_month": "No hay registros en {c} para {m}.", "head": "{n} registro(s) en {c}:", "no_record": "no existe el registro '{id}'.",
           "noted": "Nota guardada en {id}.", "note": "Nota", "total_all": "Total {m}: {t} EUR ({n} gasto(s))", "total_cat": "Total {cat} {m}: {t} EUR ({n} gasto(s))",
           "no_expenses": "No hay gastos para esa selección: total 0.00 EUR.", "bad_month": "el mes se escribe AAAA-MM, por ejemplo 2026-09.", "empty_note": "la nota está vacía.",
           "person": "Persona", "place": "Lugar", "carrier": "Transportista", "status": "Estado", "date": "Fecha", "amount": "Importe", "payer": "Pagado por", "category": "Categoría",
           "concept": "Concepto", "item": "Artículo", "qty": "Cantidad"},
}

TOOLS_TEXT: dict[str, dict[str, Any]] = {
    "family_search": {
        "en": ("Search the household records by words: expenses, appointments, shipments or pantry. Returns the matching records with their id. An empty query lists the collection.",
               {"collection": "Which records to search.", "query": "Words to look for in the records. Use the language of the records.", "month": "Only records dated in this month, written YYYY-MM."}),
        "es": ("Busca por palabras en los registros del hogar: gastos, citas, envíos o despensa. Devuelve los registros que coinciden, con su id. Una consulta vacía lista toda la colección.",
               {"collection": "En qué registros buscar.", "query": "Palabras que buscar en los registros. Usa el idioma de los registros.", "month": "Solo registros con fecha en este mes, escrito AAAA-MM."})},
    "family_get": {
        "en": ("Read one household record by its id, with its note if it has one.", {"record_id": "Record id, for example G-101."}),
        "es": ("Lee un registro del hogar por su id, con su nota si la tiene.", {"record_id": "Id del registro, por ejemplo G-101."})},
    "family_total": {
        "en": ("Add up the household expenses of a month, optionally of one category.", {"month": "Month written YYYY-MM.", "category": "Only this expense category."}),
        "es": ("Suma los gastos del hogar de un mes, si se quiere de una sola categoría.", {"month": "Mes escrito AAAA-MM.", "category": "Solo esta categoría de gasto."})},
    "family_note": {
        "en": ("Save a note on a household record.", {"record_id": "Record id.", "note": "The note to save."}),
        "es": ("Guarda una nota en un registro del hogar.", {"record_id": "Id del registro.", "note": "La nota que guardar."})},
}


def specs(lang: str) -> dict[str, ToolSpec]:
    def spec(name: str, props: dict[str, dict[str, Any]], required: list[str]) -> ToolSpec:
        desc, pdesc = TOOLS_TEXT[name][lang]
        return ToolSpec(name, desc, {"type": "object", "properties": {k: {**v, "description": pdesc[k]} for k, v in props.items()}, "required": required})
    return {
        "family_search": spec("family_search", {"collection": {"type": "string", "enum": list(COLLECTIONS)}, "query": {"type": "string"}, "month": {"type": "string"}}, ["collection"]),
        "family_get": spec("family_get", {"record_id": {"type": "string"}}, ["record_id"]),
        "family_total": spec("family_total", {"month": {"type": "string"}, "category": {"type": "string", "enum": list(EXPENSE_CATEGORIES)}}, ["month"]),
        "family_note": spec("family_note", {"record_id": {"type": "string"}, "note": {"type": "string"}}, ["record_id", "note"]),
    }


def _words_of(collection: str, r: dict[str, Any], lang: str) -> str:
    def lab(kind: str, key: str) -> str:
        e = CANON[kind][key]
        return " ".join([str(e[lang]), *e["alias"][lang]])           # type: ignore[index,list-item]
    if collection == "expenses":
        return " ".join([r["date"], lab("expense_category", r["category"]), lab("concept", r["concept"]), r["payer"], f"{r['amount']:.2f}"])
    if collection == "appointments":
        return " ".join([r["date"], r["time"], r["person"], lab("appointment", r["title"]), r["place"]])
    if collection == "shipments":
        return " ".join([lab("parcel", r["item"]), r["carrier"], r["status"], r["date"]])
    return " ".join([lab("pantry", r["item"]), str(r["qty"]), r["date"]])


_INDEX: dict[tuple[str, str], Bm25] = {}


def index(collection: str, lang: str) -> Bm25:
    key = (collection, lang)
    if key not in _INDEX:
        _INDEX[key] = Bm25([_words_of(collection, r, lang) for r in DATA[collection]])
    return _INDEX[key]


def render(collection: str, r: dict[str, Any], lang: str) -> str:
    t = lambda k: label("term", k, lang)  # noqa: E731
    if collection == "expenses":
        return f"{r['id']} | {r['date']} | {label('expense_category', r['category'], lang)} | {r['amount']:.2f} EUR | {r['payer']} | {label('concept', r['concept'], lang)}"
    if collection == "appointments":
        return f"{r['id']} | {r['date']} {r['time']} | {r['person']} | {label('appointment', r['title'], lang)} | {r['place']}"
    if collection == "shipments":
        return f"{r['id']} | {label('parcel', r['item'], lang)} | {r['carrier']} | {r['status']} | {t('eta')} {r['date']}"
    return f"{r['id']} | {label('pantry', r['item'], lang)} | {r['qty']} {label('unit', r['unit'], lang)} | {t('expires')} {r['date']}"


class FamilyTools:
    lang: str
    notes: dict[str, str]
    _empty: bool

    def _t_family_search(self, collection: str, query: str = "", month: str = "") -> str:
        m = MSG[self.lang]
        if collection not in DATA:
            raise _ToolFail(m["no_record"].format(id=collection))
        rows = DATA[collection]
        order = index(collection, self.lang).rank(query) if str(query).strip() else list(range(len(rows)))
        chosen = [rows[i] for i in order]
        if str(month).strip():
            chosen = [r for r in chosen if r["date"].startswith(str(month).strip())]
        chosen = chosen[:LIMIT]
        if not chosen:
            self._empty = True
            return m["none"].format(q=query, c=collection) if str(query).strip() else m["none_month"].format(c=collection, m=month)
        return m["head"].format(n=len(chosen), c=collection) + "\n" + "\n".join(render(collection, r, self.lang) for r in chosen)

    def _t_family_get(self, record_id: str) -> str:
        m = MSG[self.lang]
        found = BY_ID.get(str(record_id).strip().upper())
        if found is None:
            raise _ToolFail(m["no_record"].format(id=record_id))
        collection, r = found
        text = render(collection, r, self.lang)
        return text + (f"\n{m['note']}: {self.notes[r['id']]}" if r["id"] in self.notes else "")

    def _t_family_total(self, month: str, category: str = "") -> str:
        m = MSG[self.lang]
        month = str(month).strip()
        if len(month) != 7 or month[4] != "-" or not (month[:4] + month[5:]).isdigit():
            raise _ToolFail(m["bad_month"])
        rows = [r for r in EXPENSES if r["date"].startswith(month) and (not category or r["category"] == category)]
        if not rows:
            return m["no_expenses"]
        total = f"{sum(r['amount'] for r in rows):.2f}"
        if category:
            return m["total_cat"].format(cat=label("expense_category", category, self.lang), m=month, t=total, n=len(rows))
        return m["total_all"].format(m=month, t=total, n=len(rows))

    def _t_family_note(self, record_id: str, note: str) -> str:
        m = MSG[self.lang]
        rid = str(record_id).strip().upper()
        if rid not in BY_ID:
            raise _ToolFail(m["no_record"].format(id=record_id))
        if not str(note).strip():
            raise _ToolFail(m["empty_note"])
        self.notes[rid] = str(note).strip()
        return m["noted"].format(id=rid)


__all__ = ["BY_ID", "COLLECTIONS", "DATA", "FamilyTools", "MSG", "index", "render", "specs"]
