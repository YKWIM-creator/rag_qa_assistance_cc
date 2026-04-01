from urllib.parse import urlparse

_URL_SIGNALS = {
    "admissions": ["/admissions", "/apply", "/requirements", "/enrollment", "/undergraduate", "/graduate"],
    "academics": ["/academics", "/programs", "/majors", "/departments", "/courses", "/curriculum", "/degrees"],
    "financial_aid": ["/financial", "/aid", "/scholarships", "/tuition", "/fees", "/grants", "/loans"],
    "student_services": ["/housing", "/advising", "/health", "/career", "/clubs", "/student-life", "/student-affairs"],
}

_H1_SIGNALS = {
    "admissions": ["admissions", "apply", "requirements", "enrollment"],
    "academics": ["academics", "programs", "majors", "departments", "courses", "curriculum"],
    "financial_aid": ["financial", "aid", "scholarship", "tuition", "fees", "grant"],
    "student_services": ["housing", "advising", "health", "career", "clubs"],
}


def classify_page(url: str, h1_text: str) -> str:
    """Classify a page into one of 5 content types based on URL and h1 text."""
    path = urlparse(url.lower()).path
    h1 = h1_text.lower()

    for page_type, signals in _URL_SIGNALS.items():
        for signal in signals:
            if signal in path:
                return page_type

    for page_type, signals in _H1_SIGNALS.items():
        for signal in signals:
            if signal in h1:
                return page_type

    return "general"
