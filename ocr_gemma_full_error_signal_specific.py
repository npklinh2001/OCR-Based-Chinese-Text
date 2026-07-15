from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path

from cerebras.cloud.sdk import Cerebras

MODEL = "gemma-4-31b"

API_KEYS = [
    #     "csk-6jrvcmmdp58vmrxfmc4tn3kr5ph3wn66pt2r35dw23xr9pkr", # npkl01112001
    #     "csk-y233medkv2m29x3dfhdxd6jpnhwcnfxjeh4td3pkymhjecww", #jamejordan
    # "csk-c8e235mwe8mp5tx4nrn53xhymwx3v2kt4crckfd986eh9r9p", #lekhanhphuong
    # "csk-6cv3hk46vv4wj8eh2j5j4df949mkt82n65mfkp9pk59532hp", # nguyenphankhanhlinh2001
    # "csk-trkjhp6hxc3yrxcmmvcdpde4wm5x5penpf8y5e6r3jkdwnfh",
    "csk-nnnjxj6cpvvrkevrvchnm6rfe9ren99e9rph5h5jrepe3hv9"

    


]

OCR_PROMPT = (
    "OCR this image. Return only the text visible in the image. "
    "Preserve line breaks. Do not translate, explain, or add notes. "
    "For traditional Chinese or Han documents written vertically, "
    "Read from the rightmost column first, read each column from top to bottom, "
    "then continue to the next column on the left. "
    "If the text is Han or Chinese characters, keep the original characters."
)


def download_image(image_url: str):
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


def call_cerebras(image_url: str, api_key: str):
    image_bytes, mime_type = download_image(image_url)

    client = Cerebras(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def guess_doc_id(image_url: str) -> str:
    """
    Suy ra "mã tài liệu" từ tên file ảnh trong URL, để tìm đúng file JSON.
    Ví dụ: .../nlvnpf-0250-005.jpg -> "nlvnpf-0250"
    (bỏ số trang "-005" ở cuối nếu có).
    """
    stem = Path(image_url.split("?")[0]).stem  # nlvnpf-0250-005
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]  # nlvnpf-0250
    return stem


def find_json_for_url(image_url: str) -> Path | None:
    """Tìm file JSON khớp với mã tài liệu suy ra từ URL."""
    doc_id = guess_doc_id(image_url)
    matches = sorted(Path(".").glob(f"ocr_gemma-4-31b_*{doc_id}*_results.json"))
    if matches:
        return matches[0]
    return None


def save_result_to_json(image_url: str, result: str, json_path: Path | None = None):
    """
    Ghi/append kết quả OCR vào đúng file JSON tương ứng với URL.
    - Nếu file đã có item cùng url -> cập nhật result.
    - Nếu chưa có item đó -> thêm mới vào cuối danh sách.
    - Nếu không tìm thấy file phù hợp -> tạo file mới theo mã tài liệu.
    """
    if json_path is None:
        json_path = find_json_for_url(image_url)

    if json_path is None:
        doc_id = guess_doc_id(image_url)
        json_path = Path(f"ocr_gemma-4-31b_{doc_id}_results.json")
        print(f"⚠️ Không tìm thấy file JSON có sẵn, sẽ tạo mới: {json_path.name}")
        data = []
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    found = False
    for item in data:
        if item.get("url") == image_url:
            item["result"] = result
            found = True
            break

    if not found:
        data.append({"url": image_url, "result": result})
        print(f"➕ Không thấy item khớp url trong file, đã thêm item mới vào {json_path.name}")
    else:
        print(f"✏️  Đã cập nhật result cho item khớp url trong {json_path.name}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Đã lưu vào: {json_path.resolve()}")


def run_single_url(image_url: str, api_key: str | None = None, save: bool = True):
    """Chạy OCR cho đúng 1 link ảnh, in kết quả, và (mặc định) lưu vào đúng file JSON."""
    key = api_key or API_KEYS[0]
    print(f"Đang OCR: {image_url}")
    try:
        result = call_cerebras(image_url, key)
        print("\n===== KẾT QUẢ OCR =====")
        print(result)

        if save:
            save_result_to_json(image_url, result)

        return result
    except Exception as e:
        print(f"❌ Lỗi OCR: {e}")
        return None


def retry_json(json_path: Path):
    print(f"\n===== {json_path.name} =====")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lọc ra những item nào thực sự chứa ký tự �
    invalid_indices = [
        idx for idx, item in enumerate(data) if "�" in item.get("result", "")
    ]

    if not invalid_indices:
        print("Không có ký tự � -> Bỏ qua.")
        return

    print(f"Phát hiện {len(invalid_indices)}/{len(data)} item chứa ký tự � -> Chỉ OCR lại các item này.")

    key_index = 0
    success_count = 0

    for i, idx in enumerate(invalid_indices, start=1):
        item = data[idx]
        print(f"[{i}/{len(invalid_indices)}] {item['url']}")

        try:
            result = call_cerebras(
                item["url"],
                API_KEYS[key_index],
            )

            item["result"] = result
            success_count += 1

            print(f"    ✅ OCR thành công")

            # OCR thành công thì lưu luôn
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    ❌ Lỗi OCR: {e}")
            # Giữ nguyên result cũ, không thêm trường error

        key_index = (key_index + 1) % len(API_KEYS)

    print(f"Hoàn thành: {success_count}/{len(invalid_indices)} item OCR thành công.")

def count_invalid(json_files):
    """Đếm tổng số item còn chứa ký tự � trên toàn bộ các file."""
    total = 0
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        total += sum(1 for item in data if "�" in item.get("result", ""))
    return total


def main(max_rounds: int = 20):
    json_files = sorted(Path(".").glob("ocr_gemma-4-31b_*_results.json"))

    print(f"Tìm thấy {len(json_files)} file.\n")

    round_num = 0
    previous_invalid_count = None

    while True:
        round_num += 1
        print(f"\n========== VÒNG {round_num} ==========")

        files_need_retry = []
        total_invalid_items = 0

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            invalid_count = sum(1 for item in data if "�" in item.get("result", ""))

            if invalid_count > 0:
                print(f"[CÓ]    {json_file.name}  ({invalid_count} item lỗi)")
                files_need_retry.append(json_file)
                total_invalid_items += invalid_count
            else:
                print(f"[KHÔNG] {json_file.name}")

        print(
            f"\nTổng cộng: {len(files_need_retry)}/{len(json_files)} file cần OCR lại, "
            f"tương ứng {total_invalid_items} item bị lỗi.\n"
        )

        if not files_need_retry:
            print("🎉 Không còn file nào chứa ký tự lỗi. Hoàn tất!")
            break

        # Chỉ OCR lại những file còn lỗi
        for json_file in files_need_retry:
            retry_json(json_file)

        current_invalid_count = count_invalid(json_files)
        print(f"\nSố ảnh còn lỗi sau vòng {round_num}: {current_invalid_count}")

        if current_invalid_count == 0:
            print("🎉 Không còn ký tự lỗi nào. Hoàn tất!")
            break

        if previous_invalid_count is not None and current_invalid_count >= previous_invalid_count:
            print(
                "⚠️ Số lỗi không giảm so với vòng trước "
                f"({previous_invalid_count} -> {current_invalid_count}). "
                "Có thể do ảnh/link bị lỗi vĩnh viễn. Dừng lại để tránh lặp vô hạn."
            )
            break

        previous_invalid_count = current_invalid_count

        if round_num >= max_rounds:
            print(f"⚠️ Đã đạt giới hạn {max_rounds} vòng. Dừng lại.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR ảnh bằng Cerebras Gemma")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Chạy OCR cho đúng 1 link ảnh cụ thể, thay vì quét các file JSON.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Số vòng lặp tối đa khi chạy chế độ quét file JSON (mặc định 20).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Chỉ in kết quả ra màn hình, không ghi vào file JSON (dùng với --url).",
    )
    args = parser.parse_args()

    if args.url:
        run_single_url(args.url, save=not args.no_save)
    else:
        main(max_rounds=args.max_rounds)