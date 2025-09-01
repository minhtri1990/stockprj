"""
Module: load_all_symbols.py
Chức năng:
  - Đọc danh sách mã cổ phiếu từ file CSV (mặc định: data/symbols_vn.csv)
  - (Tuỳ chọn) Lấy danh sách từ API vnstock để:
      + overwrite (from_api=True, merge_api=False)
      + merge/gộp (merge_api=True)
  - Nếu không có file & API lỗi -> dùng fallback list.

Hàm chính:
  load_all_symbols(
      from_api: bool = False,
      merge_api: bool = False,
      symbol_file: Optional[Path|str] = None,
      fallback: Optional[List[str]] = None,
      validate: bool = True,
      save_on_overwrite: bool = True,
      save_on_merge: bool = True,
      api_candidate_methods: Optional[List[str]] = None,
  ) -> Tuple[List[str], str]

Trả về:
  symbols (List[str]), note (str mô tả quá trình)

Bạn có thể bọc hàm này bằng st.cache_data trong app Streamlit:
  @st.cache_data(ttl=600)
  def cached_symbols(**kwargs):
      return load_all_symbols(**kwargs)

Ví dụ dùng trong app.py:
  from utils.load_all_symbols import load_all_symbols
  symbols, note = load_all_symbols()
  symbol = st.selectbox("Mã", symbols)
  st.caption(note)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Set
import streamlit as st

# -------------------------------------------------
# Cấu hình
# -------------------------------------------------
@dataclass
class SymbolLoaderConfig:
    symbol_file: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "symbols_vn.csv")
    fallback: List[str] = field(default_factory=lambda: [
        "SSI"
    ])
    api_candidate_methods: List[str] = field(default_factory=lambda: [
        "stock_listing", "listing", "symbols", "equities", "list_symbols"
    ])
    max_len: int = 8       # chiều dài tối đa mã
    min_len: int = 2       # chiều dài tối thiểu mã
    allow_chars: str = ""  # có thể bổ sung ký tự đặc biệt nếu cần (ví dụ: ".")


# -------------------------------------------------
# Các hàm tiện ích
# -------------------------------------------------
def _normalize_symbol(s: str, cfg: SymbolLoaderConfig) -> Optional[str]:
    if not isinstance(s, str):
          return None
    s2 = s.strip().upper()
    if not (cfg.min_len <= len(s2) <= cfg.max_len):
        return None
    # Loại bỏ ký tự không phải chữ/số (trừ khi được cho phép)
    if not all(ch.isalnum() or ch in cfg.allow_chars for ch in s2):
        return None
    return s2

def _read_csv_symbols(path: Path, cfg: SymbolLoaderConfig) -> Tuple[List[str], str]:
    notes = []
    if not path.exists():
        return [], f"File '{path.name}' chưa tồn tại."
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if "symbol" not in df.columns:
            return [], f"File '{path.name}' thiếu cột 'symbol'."
        clean: List[str] = []
        for raw in df["symbol"].dropna().tolist():
            norm = _normalize_symbol(str(raw), cfg)
            if norm:
                clean.append(norm)
        clean = sorted(set(clean))
        return clean, f"Đọc {len(clean)} mã từ file."
    except Exception as e:
        # fallback đọc thủ công bằng csv nếu pandas không có hoặc lỗi
        try:
            symbols: Set[str] = set()
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if "symbol" not in reader.fieldnames:
                    return [], f"File '{path.name}' lỗi/thiếu cột symbol: {e}"
                for row in reader:
                    raw = row.get("symbol")
                    if raw:
                        norm = _normalize_symbol(raw, cfg)
                        if norm:
                            symbols.add(norm)
            clean = sorted(symbols)
            return clean, f"Đọc {len(clean)} mã từ file (csv module)."
        except Exception as e2:
            return [], f"Lỗi đọc file '{path.name}': {e}; secondary: {e2}"

def _fetch_api_symbols(cfg: SymbolLoaderConfig) -> Tuple[List[str], str]:
    try:
        from vnstock import Vnstock
    except Exception as e:
        return [], f"Không import được vnstock: {e}"

    base = Vnstock()
    symbols: Set[str] = set()
    last_error: Optional[Exception] = None

    for m in cfg.api_candidate_methods:
        if hasattr(base, m):
            try:
                res = getattr(base, m)()
                if res is None:
                    continue
                # Res có thể là list hoặc DataFrame
                if isinstance(res, list):
                    for s in res:
                        norm = _normalize_symbol(s, cfg)
                        if norm:
                            symbols.add(norm)
                else:
                    # DataFrame
                    df = res
                    candidate_cols = ["ticker", "symbol", "code", "stock_code", "STOCK_CODE"]
                    picked = None
                    for col in candidate_cols:
                        if col in df.columns:
                            picked = col
                            break
                    if picked:
                        for s in df[picked]:
                            norm = _normalize_symbol(s, cfg)
                            if norm:
                                symbols.add(norm)
            except Exception as e:
                last_error = e
                continue

    if not symbols:
        note = "API không trả về mã."
        if last_error:
            note += f" Last error: {last_error}"
        return [], note

    ordered = sorted(symbols)
    note = f"Lấy {len(ordered)} mã từ API."
    if last_error:
        note += f" (Có lỗi sau cùng: {last_error})"
    return ordered, note

def _write_symbols(path: Path, symbols: Iterable[str]) -> Tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        df = pd.DataFrame({"symbol": sorted(set(symbols))})
        df.to_csv(path, index=False)
        return True, f"Ghi file '{path.name}' với {len(df)} mã."
    except Exception as e:
        return False, f"Lỗi ghi file '{path}': {e}"


# ================================
# Cache danh sách mã
# ================================
@st.cache_data(ttl=600, show_spinner=False)
def get_symbols():
    return load_all_symbols()

# -------------------------------------------------
# Hàm chính
# -------------------------------------------------
def load_all_symbols(
    from_api: bool = False,
    merge_api: bool = False,
    symbol_file: Optional[Path | str] = None,
    fallback: Optional[List[str]] = None,
    validate: bool = True,
    save_on_overwrite: bool = True,
    save_on_merge: bool = True,
    api_candidate_methods: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Load danh sách mã cổ phiếu.

    Tham số:
      from_api: True => bỏ qua file, lấy API (overwrite nếu save_on_overwrite)
      merge_api: True => đọc file + API rồi gộp (ưu tiên file) (ghi lại nếu save_on_merge)
      symbol_file: đường dẫn file CSV; nếu None dùng mặc định data/symbols_vn.csv
      fallback: danh sách fallback nếu cả file & API đều thất bại
      validate: bật để lọc & chuẩn hoá mã
      save_on_overwrite: khi from_api=True ghi đè file
      save_on_merge: khi merge_api=True ghi lại file sau khi gộp
      api_candidate_methods: danh sách tên phương thức thử gọi trong vnstock

    Trả về:
      (symbols: List[str], note: str)
    """
    cfg = SymbolLoaderConfig()
    if symbol_file:
        cfg.symbol_file = Path(symbol_file)
    if fallback:
        cfg.fallback = fallback
    if api_candidate_methods:
        cfg.api_candidate_methods = api_candidate_methods

    notes: List[str] = []

    # 1. Đọc từ file (trừ khi ép from_api)
    file_syms: List[str] = []
    if not from_api:
        file_syms, note_file = _read_csv_symbols(cfg.symbol_file, cfg)
        notes.append(note_file)

    # 2. Lấy từ API nếu cần (from_api hoặc merge hoặc không có file)
    api_syms: List[str] = []
    if from_api or merge_api or not file_syms:
        api_syms, note_api = _fetch_api_symbols(cfg)
        notes.append(note_api)

    # 3. Logic hợp nhất
    final_syms: List[str]
    if from_api:
        # Overwrite mode
        final_syms = api_syms if api_syms else file_syms
        if save_on_overwrite and api_syms:
            ok, note_w = _write_symbols(cfg.symbol_file, api_syms)
            notes.append(note_w)
    elif merge_api:
        merged = sorted(set(file_syms) | set(api_syms))
        final_syms = merged
        if save_on_merge and merged:
            ok, note_w = _write_symbols(cfg.symbol_file, merged)
            notes.append(note_w)
    else:
        final_syms = file_syms if file_syms else api_syms

    # 4. Fallback
    if not final_syms:
        final_syms = cfg.fallback
        notes.append(f"Fallback {len(final_syms)} mã mẫu.")

    # 5. Validate (đã validate trong lúc đọc, nhưng nếu muốn có thể re-check)
    if validate:
        cleaned = []
        seen = set()
        for s in final_syms:
            norm = _normalize_symbol(s, cfg)
            if norm and norm not in seen:
                cleaned.append(norm)
                seen.add(norm)
        final_syms = sorted(cleaned)

    note = " | ".join(notes)
    return final_syms


# -------------------------------------------------
# CLI / Test nhanh
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load danh sách cổ phiếu từ file/API.")
    parser.add_argument("--from-api", action="store_true", help="Bỏ qua file, lấy API & overwrite (nếu có).")
    parser.add_argument("--merge-api", action="store_true", help="Gộp API vào file.")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn file CSV tuỳ chỉnh.")
    parser.add_argument("--no-save-overwrite", action="store_true", help="Không ghi file khi from_api.")
    parser.add_argument("--no-save-merge", action="store_true", help="Không ghi file khi merge_api.")
    args = parser.parse_args()

    symbols, note = load_all_symbols(
        from_api=args.from_api,
        merge_api=args.merge_api,
        symbol_file=args.file,
        save_on_overwrite=not args.no_save_overwrite,
        save_on_merge=not args.no_save_merge,
    )
    print(f"Số mã: {len(symbols)}")
    print("Ghi chú:", note)
    print("Ví dụ 20 mã đầu:", symbols[:20])
