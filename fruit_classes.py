"""Agrupa las variedades de Fruits-360 en tipos de fruta para la clasificación."""

FRUIT_FAMILIES = (
    "Apple", "Apricot", "Avocado", "Banana", "Blackberry",
    "Blueberry", "Cactus fruit", "Cantaloupe", "Carambola", "Cherry Wax", "Cherry",
    "Cherimoya", "Clementine", "Cocos", "Dates", "Fig", "Gooseberry",
    "Granadilla", "Grape", "Grapefruit", "Guava", "Huckleberry", "Kaki",
    "Kiwi", "Kumquats", "Lemon", "Limes", "Lychee", "Mandarine",
    "Mango", "Mangostan", "Maracuja", "Melon", "Mulberry", "Nectarine",
    "Orange", "Papaya", "Passion fruit", "Peach", "Pear", "Pepino",
    "Physalis", "Pineapple", "Pitahaya", "Plum", "Pomegranate", "Pomelo",
    "Quince", "Rambutan", "Raspberry", "Redcurrant", "Salak",
    "Strawberry", "Tamarillo", "Tangelo", "Watermelon",
)

SPANISH_NAMES = {
    "Apple": "Manzana", "Apricot": "Damasco", "Avocado": "Palta",
    "Banana": "Banana", "Blackberry": "Mora", "Blueberry": "Arándano",
    "Cactus fruit": "Higo chumbo", "Cantaloupe": "Melón cantalupo",
    "Carambola": "Carambola", "Cherry Wax": "Cereza de Java",
    "Cherry": "Cereza", "Cherimoya": "Chirimoya", "Clementine": "Clementina",
    "Cocos": "Coco", "Dates": "Dátil", "Fig": "Higo",
    "Gooseberry": "Grosella espinosa", "Granadilla": "Granadilla",
    "Grape": "Uva", "Grapefruit": "Pomelo", "Guava": "Guayaba",
    "Huckleberry": "Arándano silvestre", "Kaki": "Caqui", "Kiwi": "Kiwi",
    "Kumquats": "Kumquat", "Lemon": "Limón", "Limes": "Lima",
    "Lychee": "Lichi", "Mandarine": "Mandarina", "Mango": "Mango",
    "Mangostan": "Mangostán", "Maracuja": "Maracuyá", "Melon": "Melón",
    "Mulberry": "Mora de morera", "Nectarine": "Nectarina",
    "Orange": "Naranja", "Papaya": "Papaya", "Passion fruit": "Fruta de la pasión",
    "Peach": "Durazno", "Pear": "Pera", "Pepino": "Pepino dulce",
    "Physalis": "Fisalis", "Pineapple": "Ananá", "Pitahaya": "Pitahaya",
    "Plum": "Ciruela", "Pomegranate": "Granada", "Pomelo": "Pomelo dulce",
    "Quince": "Membrillo", "Rambutan": "Rambután", "Raspberry": "Frambuesa",
    "Redcurrant": "Grosella roja", "Salak": "Salak",
    "Strawberry": "Frutilla", "Tamarillo": "Tomate de árbol",
    "Tangelo": "Tangelo", "Watermelon": "Sandía",
}


def family_of(category: str) -> str | None:
    # Prefijos largos primero: Cherry Wax no debe quedar como Cherry.
    normalized = category.casefold()
    for family in sorted(FRUIT_FAMILIES, key=len, reverse=True):
        prefix = family.casefold()
        if normalized == prefix or normalized.startswith(prefix + " "):
            return family
    return None


def choose_categories(training_root, test_root) -> list[str]:
    return sorted(
        path.name for path in training_root.iterdir()
        if path.is_dir() and family_of(path.name) and (test_root / path.name).is_dir()
    )
