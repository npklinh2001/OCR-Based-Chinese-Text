from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.request
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "op/x-ai/grok-4.3"
BASE_URL = "https://api.vilao.ai/v1"

EXCEL_PATH = "DS Đề tài KHDL.xlsx"
SHEET_NAME = "Tách trang"
FIRST_IMAGE_URL_TEMPLATE = (
    "https://lib.nomfoundation.org/site_media/nom/{vnpf_id}/jpeg/{vnpf_id}-001.jpg"
)
OUTPUT_PATH_TEMPLATE = "ocr_grok-4.3_{vnpf_id}-001_results.json"

OCR_PROMPT = (
    "OCR this image. Return only the text visible in the image. "
    "Preserve line breaks. Do not translate, explain, or add notes. "
    "For traditional Chinese or Han documents written vertically, "
    "Read from the rightmost column first, read each column from top to bottom, then continue to the next column on the left. "
    "If the text is Han or Chinese characters, keep the original characters."
)

# Lấy API key từ biến môi trường thay vì hardcode trong source.
API_KEYS = [
    "sk-935ef973d080bd7a0b93c7e803ac5ac19246b528b0355f7f5c3e17918b4ac186",  # ngphkhanhlinh2001
]


def download_image(image_url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "ocr-nom/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        image_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")

    mime_type = content_type.split(";", 1)[0].strip()
    if not mime_type:
        mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"

    return image_bytes, mime_type


def call_vilao_for_url(image_url: str, api_key: str) -> str:
    image_bytes, mime_type = download_image(image_url)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def ocr_image_urls(
    image_urls: list[str],
    output_path: str | Path,
    ocr_func=call_vilao_for_url,
) -> list[dict[str, str]]:
    output_file = Path(output_path)
    saved_results = load_json_results(output_file)
    results = []

    key_index = 0
    for image_url in image_urls:
        try:
            print(f"Processing {image_url}...")
            result = ocr_func(image_url, api_key=API_KEYS[key_index])
            print(f"Result for {image_url}: {result[:50]}...")
            entry = {"url": image_url, "result": result}
        except Exception as exc:
            print(f"Error processing {image_url}: {exc}")
            entry = {"url": image_url, "error": str(exc)}

        results.append(entry)
        saved_results.append(entry)
        write_json_results(output_file, saved_results)

        key_index += 1
        if key_index >= len(API_KEYS):
            key_index = 0

    return results


def load_json_results(output_file: Path) -> list[dict[str, str]]:
    if not output_file.exists():
        return []

    text = output_file.read_text(encoding="utf-8").strip()
    if not text:
        return []

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"File JSON output phải là một list: {output_file}")

    return data


def write_json_results(output_file: Path, results: list[dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_image_url_range(first_image_url: str, from_page: int, to_page: int) -> list[str]:
    if from_page > to_page:
        raise ValueError("from_page phải nhỏ hơn hoặc bằng to_page.")

    page_pattern = re.compile(r"-\d{3}(\.[A-Za-z0-9]+)$")
    if not page_pattern.search(first_image_url):
        raise ValueError("URL cần kết thúc bằng dạng -001.jpg, -002.jpg, ...")

    return [
        page_pattern.sub(lambda match: f"-{page:03d}{match.group(1)}", first_image_url)
        for page in range(from_page, to_page + 1)
    ]


def build_first_image_url(vnpf_id: str) -> str:
    return FIRST_IMAGE_URL_TEMPLATE.format(vnpf_id=vnpf_id)


def build_output_path(vnpf_id: str) -> str:
    return OUTPUT_PATH_TEMPLATE.format(vnpf_id=vnpf_id)


def read_page_number(value: object, column_name: str, row_number: int) -> int:
    if pd.isna(value):
        raise ValueError(f"Dòng {row_number} thiếu giá trị cột {column_name}.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dòng {row_number} cột {column_name} không phải số trang hợp lệ: {value!r}"
        ) from exc

    if not numeric_value.is_integer():
        raise ValueError(
            f"Dòng {row_number} cột {column_name} phải là số nguyên: {value!r}"
        )

    page = int(numeric_value)
    if page < 1:
        raise ValueError(f"Dòng {row_number} cột {column_name} phải >= 1.")

    return page


def load_ocr_jobs(
    excel_path: str | Path = EXCEL_PATH,
    sheet_name: str = SHEET_NAME,
) -> list[dict[str, object]]:
    data = pd.read_excel(excel_path, sheet_name=sheet_name)
    data.columns = data.columns.astype(str).str.strip()

    required_columns = ["VNPF Id", "FROM", "TO"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Thiếu cột trong file Excel: {', '.join(missing_columns)}")

    jobs: list[dict[str, object]] = []
    for index, row in data.iterrows():
        row_number = index + 2
        if pd.isna(row["VNPF Id"]):
            continue

        vnpf_id = str(row["VNPF Id"]).strip()
        if not vnpf_id:
            continue

        from_page = read_page_number(row["FROM"], "FROM", row_number)
        to_page = read_page_number(row["TO"], "TO", row_number)

        jobs.append(
            {
                "vnpf_id": vnpf_id,
                "first_image_url": build_first_image_url(vnpf_id),
                "from_page": from_page,
                "to_page": to_page,
                "output_path": build_output_path(vnpf_id),
            }
        )

    return jobs


def main() -> None:
    if not API_KEYS[0]:
        raise SystemExit("Thiếu API key trong danh sách API_KEYS.")

    jobs = load_ocr_jobs()
    print(f"Đọc được {len(jobs)} dòng cần OCR từ {EXCEL_PATH}")

    for job in jobs:
        vnpf_id = str(job["vnpf_id"])
        first_image_url = str(job["first_image_url"])
        from_page = int(job["from_page"])
        to_page = int(job["to_page"])
        output_path = str(job["output_path"])

        print(f"Đang OCR {vnpf_id}: trang {from_page}-{to_page}")
        image_urls = build_image_url_range(first_image_url, from_page, to_page)
        results = ocr_image_urls(image_urls, output_path)
        print(f"Đã lưu {len(results)} kết quả vào {output_path}")


if __name__ == "__main__":
    main()