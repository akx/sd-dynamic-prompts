import collections
import json
import re
import sys

import requests

root = "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles"
styles = {
    "diva": f"{root}/sdxl_styles_diva.json",
    "fooocus": f"{root}/sdxl_styles_fooocus.json",
    "marck3nt3l": f"{root}/sdxl_styles_marc_k3nt3l.json",
    "mre": f"{root}/sdxl_styles_mre.json",
    "sai": f"{root}/sdxl_styles_sai.json",
    "twri": f"{root}/sdxl_styles_twri.json",
}

ELLIPSIS = "\u2026"  # "…" (HORIZONTAL ELLIPSIS)


def download_data():
    with requests.Session() as sess:

        def _get(url):
            resp = sess.get(url)
            resp.raise_for_status()
            print(url, file=sys.stderr)
            return resp.json()

        return {name: _get(url) for name, url in styles.items()}


def process_prompt(prompt: str) -> str:
    return prompt.replace("{prompt}", ELLIPSIS)


def main():
    pantry = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for category_name, items in download_data().items():
        if "fooocus" not in category_name:
            pantry_category_name = f"fooocus-{category_name}"
        else:
            pantry_category_name = category_name
        for item in items:
            name = re.sub(r"\s+", "-", item["name"].lower())
            name = name.removeprefix(category_name).strip("-")

            prompt = item.get("prompt")
            if prompt:
                lst = pantry[pantry_category_name][name]
                lst.append(process_prompt(prompt))
            negative_prompt = item.get("negative_prompt")
            if negative_prompt:
                lst = pantry[f"negative-{pantry_category_name}"][name]
                lst.append(process_prompt(negative_prompt))
            n += 1
    print(json.dumps(pantry, indent=2, sort_keys=True, ensure_ascii=False))
    print("=>", n, "presets", file=sys.stderr)


if __name__ == "__main__":
    main()
