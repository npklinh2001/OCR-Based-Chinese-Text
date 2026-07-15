import json
import re
from pathlib import Path

# Regex: 3 dấu cách liên tiếp trở lên (dấu cách thường " " hoặc dấu cách toàn góc "　")
# -> gộp thành 1 dấu cách thường. Không đụng vào ký tự xuống dòng \n.
EXTRA_SPACE_PATTERN = re.compile(r"[ \u3000]{3,}")

# Ký tự lỗi encoding (replacement character)
BROKEN_CHAR_PATTERN = re.compile(r"\uFFFD")


def clean_extra_spaces(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = [EXTRA_SPACE_PATTERN.sub(" ", line) for line in lines]
    return "\n".join(cleaned_lines)


def remove_broken_chars(text: str) -> str:
    return BROKEN_CHAR_PATTERN.sub("", text)


def clean_json_file(json_path: Path) -> int:
    """Trả về số item đã được chỉnh sửa trong file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed_count = 0

    for item in data:
        original = item.get("result", "")
        cleaned = clean_extra_spaces(original)
        cleaned = remove_broken_chars(cleaned)

        if cleaned != original:
            item["result"] = cleaned
            changed_count += 1

    if changed_count > 0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return changed_count


def main():
    json_files = sorted(Path(".").glob("ocr_gemma-4-31b_*_results.json"))

    print(f"Tìm thấy {len(json_files)} file.\n")

    total_changed = 0

    for json_file in json_files:
        changed = clean_json_file(json_file)
        total_changed += changed

        if changed > 0:
            print(f"[SỬA] {json_file.name}: {changed} item được dọn khoảng trắng / xóa ký tự lỗi.")
        else:
            print(f"[OK]  {json_file.name}: không có gì cần chỉnh.")

    print(f"\nHoàn tất. Tổng cộng {total_changed} item được chỉnh sửa.")


if __name__ == "__main__":
    main()