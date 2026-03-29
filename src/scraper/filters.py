from urllib.parse import urlparse

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".avi", ".mov",
}

SKIP_PATH_FRAGMENTS = [
    "/login", "/logout", "/signin", "/signout",
    "/calendar", "/events",
    "/wp-json", "/feed",
]

SKIP_QUERY_PARAMS = ["print", "sort", "filter", "page"]


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    path = parsed.path

    # Check file extension
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True

    # Check path fragments
    for fragment in SKIP_PATH_FRAGMENTS:
        if fragment in path:
            return True

    # Check query params
    query = parsed.query
    for param in SKIP_QUERY_PARAMS:
        if f"{param}=" in query:
            return True

    return False
