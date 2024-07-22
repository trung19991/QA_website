import json

def extract_code_cells(ipynb_path, output_path=None):
    # Đọc nội dung của tệp .ipynb
    with open(ipynb_path, "r", encoding="utf-8") as file:
        notebook_content = json.load(file)

    # Trích xuất các ô mã (code cells)
    code_cells = []
    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code":
            code_cells.append("".join(cell["source"]))

    # Nếu có output_path, lưu các đoạn mã vào tệp .py
    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            for code in code_cells:
                file.write(code)
                file.write("\n\n")
    else:
        # In các đoạn mã ra màn hình
        for code in code_cells:
            print(code)
            print("\n\n")

if __name__ == "__main__":
    # Đường dẫn tới tệp .ipynb
    ipynb_path = "xlm_roberta_07062024.ipynb"

    # Đường dẫn để lưu tệp .py (có thể thay đổi, hoặc để None để in ra màn hình)
    output_path = "xlm_roberta_07062024sa.py"

    # Gọi hàm để trích xuất các ô mã
    extract_code_cells(ipynb_path, output_path)
