import pandas as pd
import json
import csv
from typing import List, Dict, Any, Union

def read_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu từ file CSV và chuyển đổi thành định dạng DATABASE
    
    Cấu trúc CSV kỳ vọng:
    - Cột "items": danh sách các mặt hàng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "quantities": số lượng tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "profits": lợi nhuận tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "probabilities": xác suất tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    """
    data = []
    
    try:
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            record = {
                "items": parse_list_field(row["items"]),
                "quantities": parse_list_field(row["quantities"], convert_to=int),
                "profits": parse_list_field(row["profits"], convert_to=float),
                "probabilities": parse_list_field(row["probabilities"], convert_to=float)
            }
            data.append(record)
            
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        
    return data

def read_excel_data(file_path: str, sheet_name: str = None) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu từ file Excel và chuyển đổi thành định dạng DATABASE
    
    Cấu trúc Excel kỳ vọng:
    - Cột "items": danh sách các mặt hàng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "quantities": số lượng tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "profits": lợi nhuận tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    - Cột "probabilities": xác suất tương ứng, phân tách bằng dấu phẩy hoặc dấu chấm phẩy
    """
    data = []
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        for _, row in df.iterrows():
            record = {
                "items": parse_list_field(row["items"]),
                "quantities": parse_list_field(row["quantities"], convert_to=int),
                "profits": parse_list_field(row["profits"], convert_to=float),
                "probabilities": parse_list_field(row["probabilities"], convert_to=float)
            }
            data.append(record)
            
    except Exception as e:
        print(f"Lỗi khi đọc file Excel: {e}")
        
    return data

def read_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu từ file JSON và chuyển đổi thành định dạng DATABASE
    
    Cấu trúc JSON kỳ vọng:
    [
        {
            "items": ["A", "B", "(CD)"],
            "quantities": [2, 1, 3],
            "profits": [6, 5, 9],
            "probabilities": [0.8, 0.75, 0.6]
        },
        ...
    ]
    """
    data = []
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        
    return data

def parse_list_field(field_value: str, separator: str = ',', convert_to = None) -> List:
    """
    Chuyển đổi chuỗi phân tách thành danh sách
    Ví dụ: "A,B,(CD)" -> ["A", "B", "(CD)"]
    """
    if isinstance(field_value, list):
        return field_value
    
    items = [item.strip() for item in str(field_value).split(separator)]
    
    if convert_to:
        try:
            return [convert_to(item) for item in items]
        except:
            # Thử phân tách bằng dấu chấm phẩy nếu dấu phẩy không hoạt động
            items = [item.strip() for item in str(field_value).split(';')]
            return [convert_to(item) for item in items]
    
    return items

def read_txt_format_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu từ file txt với định dạng đặc biệt:
    items : sum utility : profits : quantities : probabilities
    
    Ví dụ:
    1 : 2:2 : 6 : 0.54
    1 : 2:2 : 2 : 0.86
    1 (23,12) : 5:2 2.0 : 4 5 : 0.2 0.98
    
    Returns:
    --------
    List[Dict[str, Any]]
        Dữ liệu ở định dạng DATABASE
    """
    data = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:  # Bỏ qua dòng trống
                continue
                
            # Phân tách các phần của dòng dữ liệu
            parts = line.split(':')
            
            if len(parts) < 4:
                print(f"Bỏ qua dòng không đúng định dạng: {line}")
                continue
                
            # Xử lý phần items
            items_part = parts[0].strip()
            items = []
            
            # Xử lý trường hợp có dạng "1 (23,12)"
            items_elements = items_part.split()
            for item in items_elements:
                if item.startswith('(') and item.endswith(')'):
                    items.append(item)
                else:
                    items.append(item)
            
            # Xử lý phần profits
            profits_part = parts[2].strip()
            profits = [float(p.strip()) for p in profits_part.split()]

            # Xử lý phần quantities
            quantities_part = parts[3].strip()
            quantities = [float(q.strip()) for q in quantities_part.split()]
            
            # Xử lý phần probabilities
            probabilities_part = parts[4].strip()
            probabilities = [float(p.strip()) for p in probabilities_part.split()]
            
            # Tạo bản ghi dữ liệu
            record = {
                "items": items,
                "quantities": quantities,
                "profits": profits,
                "probabilities": probabilities
            }
            
            data.append(record)
            
    except Exception as e:
        print(f"Lỗi khi đọc file txt: {e}")
    
    return data

def read_merged_data(file_path: str, file_type: str = None) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu từ nguồn hợp nhất và chuyển đổi thành định dạng DATABASE
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file dữ liệu
    file_type : str, optional
        Loại file ('csv', 'excel', 'json', 'txt'). Nếu không được chỉ định, sẽ được suy ra từ phần mở rộng của file.
        
    Returns:
    --------
    List[Dict[str, Any]]
        Dữ liệu ở định dạng DATABASE
    """
    if file_type is None:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension in ['csv']:
            file_type = 'csv'
        elif file_extension in ['xls', 'xlsx']:
            file_type = 'excel'
        elif file_extension in ['json']:
            file_type = 'json'
        elif file_extension in ['txt']:
            file_type = 'txt'
        else:
            raise ValueError(f"Không thể xác định loại file từ phần mở rộng '{file_extension}'. Vui lòng chỉ định file_type.")
    
    if file_type == 'csv':
        return read_csv_data(file_path)
    elif file_type == 'excel':
        return read_excel_data(file_path)
    elif file_type == 'json':
        return read_json_data(file_path)
    elif file_type == 'txt':
        return read_txt_format_data(file_path)
    else:
        raise ValueError(f"Loại file không được hỗ trợ: {file_type}. Các loại được hỗ trợ: 'csv', 'excel', 'json', 'txt'")

# Ví dụ sử dụng:
if __name__ == "__main__":
    # Đọc dữ liệu từ file (thay đổi đường dẫn tương ứng)
    # data = read_merged_data("path/to/your/data.csv")
    # data = read_merged_data("path/to/your/data.xlsx")
    data = read_merged_data("/home/loc-dang/Projects/bayesian-network/data/merged_prob_accidents_utility_spmf.txt")
    
    # In dữ liệu để kiểm tra
    print(data)
    
    # Sử dụng dữ liệu với BayesianMiner
    # from bayes_miner import BayesianMiner
    # from helper import create_utility_dict
    # bayes_miner = BayesianMiner(create_utility_dict(data), top_k=5, min_sup=0.5)
    # bayes_miner.run()
    # print(bayes_miner.get_top_k_candidates())