from app.books.metadata import infer_reading_type, map_categories



def test_category_mapping_to_allowed_values_only():
    values = ["Epic Fantasy", "Science Fiction", "Self-Help", "Cooking"]
    assert map_categories(values) == ["Fantasy", "Sci-Fi", "Self-development"]



def test_reading_type_mapping():
    assert infer_reading_type("This is an audiobook edition") == "Audiobook"
    assert infer_reading_type("DRM-free ebook format") == "eBook"
    assert infer_reading_type("hardcover") == "Physical"
