import pytest
from src.scraper.classifier import classify_page


@pytest.mark.parametrize("url,h1,expected", [
    ("https://baruch.cuny.edu/admissions/apply", "Apply Now", "admissions"),
    ("https://baruch.cuny.edu/admissions/requirements", "Requirements", "admissions"),
    ("https://hunter.cuny.edu/academics/programs", "Programs", "academics"),
    ("https://hunter.cuny.edu/departments/math", "Math Department", "academics"),
    ("https://qc.cuny.edu/financial-aid/scholarships", "Scholarships", "financial_aid"),
    ("https://qc.cuny.edu/tuition", "Tuition & Fees", "financial_aid"),
    ("https://brooklyn.cuny.edu/housing", "Student Housing", "student_services"),
    ("https://brooklyn.cuny.edu/student-affairs/advising", "Academic Advising", "student_services"),
    ("https://baruch.cuny.edu/about", "About Baruch", "general"),
    ("https://baruch.cuny.edu/news/2024", "News", "general"),
])
def test_classify_page(url, h1, expected):
    assert classify_page(url, h1) == expected


def test_classify_page_empty_h1():
    result = classify_page("https://baruch.cuny.edu/admissions", "")
    assert result == "admissions"


def test_classify_page_no_signals():
    result = classify_page("https://baruch.cuny.edu/contact", "Contact Us")
    assert result == "general"
