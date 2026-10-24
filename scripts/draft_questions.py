"""Generate draft evaluation questions for golden dataset.

Outputs questions to evaluation/golden_dataset.jsonl with:
- question (including conversational student phrasing)
- group (tín chỉ, học vụ, thi-điểm, tốt nghiệp, câu khó)
- relevant [{source_file, page, article}]
- verified: false
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation"
OUTPUT_FILE = EVAL_DIR / "golden_dataset.jsonl"

# 105 meticulously curated evaluation questions covering all 5 groups and all 11 PDF files
DRAFT_ITEMS = [
    # =========================================================================
    # NHÓM 1: TÍN CHỈ (20 câu)
    # =========================================================================
    {
        "id": "tc_001",
        "question": "Sinh viên hệ chính quy được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ chính?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 7, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_002",
        "question": "Số tín chỉ tối thiểu mà sinh viên bình thường phải đăng ký trong một học kỳ chính là bao nhiêu?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 7, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_003",
        "question": "em bị cảnh báo học vụ thì kỳ sau được đăng ký tối đa và tối thiểu mấy tín chỉ vậy bot",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 7, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_004",
        "question": "Trong học kỳ hè sinh viên được đăng ký tối đa bao nhiêu tín chỉ?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 7, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_005",
        "question": "em lỡ đăng ký môn rồi giờ muốn rút bớt học phần thì làm sao và có được trả lại học phí ko",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 8, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "tc_006",
        "question": "Thời hạn để sinh viên xin rút bớt học phần đã đăng ký trong học kỳ chính là khi nào?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 8, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "tc_007",
        "question": "Rút học phần thì điểm môn đó trên bảng điểm sẽ hiển thị chữ gì?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 8, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "tc_008",
        "question": "Điều kiện để sinh viên được phép đăng ký học vượt là gì?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 7, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_009",
        "question": "Sinh viên có được đăng ký học cải thiện điểm đối với các môn đã đạt không?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 9, "article": "Điều 10"}],
        "verified": False,
    },
    {
        "id": "tc_010",
        "question": "học cải thiện điểm thì điểm cũ có bị xóa không hay lấy điểm nào cao hơn",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 9, "article": "Điều 10"}],
        "verified": False,
    },
    {
        "id": "tc_011",
        "question": "Nếu bị rớt môn bắt buộc thì sinh viên phải làm gì?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 9, "article": "Điều 10"}],
        "verified": False,
    },
    {
        "id": "tc_012",
        "question": "Nếu bị điểm F môn tự chọn thì có bắt buộc phải học lại đúng môn đó không?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 9, "article": "Điều 10"}],
        "verified": False,
    },
    {
        "id": "tc_013",
        "question": "Học phần tiên quyết là gì và quy định ràng buộc khi đăng ký môn học?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 5, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "tc_014",
        "question": "Học phần học trước khác gì so với học phần tiên quyết?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 5, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "tc_015",
        "question": "Học phần song hành là gì?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 5, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "tc_016",
        "question": "Một tín chỉ lý thuyết được tính tương đương với bao nhiêu giờ học trên lớp?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 4, "article": "Điều 3"}],
        "verified": False,
    },
    {
        "id": "tc_017",
        "question": "Một tín chỉ thực hành hoặc thí nghiệm yêu cầu bao nhiêu giờ thực học?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 4, "article": "Điều 3"}],
        "verified": False,
    },
    {
        "id": "tc_018",
        "question": "Để hoàn thành 1 tín chỉ, sinh viên cần dành bao nhiêu giờ tự học và chuẩn bị cá nhân?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 4, "article": "Điều 3"}],
        "verified": False,
    },
    {
        "id": "tc_019",
        "question": "Quy định về việc đăng ký học thêm môn học ngoài chương trình đào tạo của ngành?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 8, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "tc_020",
        "question": "Đăng ký trễ hạn môn học thì sinh viên phải xử lý thế nào theo quy định học vụ?",
        "group": "tín chỉ",
        "relevant": [{"source_file": "28_qd-dhcntt_10-01-2023_quy_dinh_cong_tac_hoc_vu.pdf", "page": 3, "article": None}],
        "verified": False,
    },

    # =========================================================================
    # NHÓM 2: HỌC VỤ (22 câu)
    # =========================================================================
    {
        "id": "hv_001",
        "question": "Sinh viên bị cảnh báo học vụ khi nào?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 16, "article": "Điều 18"}],
        "verified": False,
    },
    {
        "id": "hv_002",
        "question": "Tiêu chí cụ thể về điểm trung bình học kỳ để bị cảnh báo kết quả học tập là bao nhiêu?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 16, "article": "Điều 18"}],
        "verified": False,
    },
    {
        "id": "hv_003",
        "question": "Bị bao nhiêu lần cảnh báo học vụ thì sinh viên sẽ bị buộc thôi học?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 19"}],
        "verified": False,
    },
    {
        "id": "hv_004",
        "question": "Những trường hợp nào sinh viên bị nhà trường buộc thôi học?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 19"}],
        "verified": False,
    },
    {
        "id": "hv_005",
        "question": "em muốn xin tạm dừng học tập (bảo lưu) một học kỳ thì cần những điều kiện gì",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "hv_006",
        "question": "Sinh viên năm nhất học kỳ 1 có được phép xin tạm dừng học tập không?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "hv_007",
        "question": "Thời gian tạm dừng học tập tối đa trong toàn khóa học là bao lâu?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "hv_008",
        "question": "Sau khi hết thời gian tạm dừng học tập, sinh viên cần làm thủ tục gì để quay lại học tiếp?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "hv_009",
        "question": "Thời gian đào tạo tối đa cho phép để hoàn thành khóa học đại học tại UIT là mấy năm?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 4, "article": "Điều 2"}],
        "verified": False,
    },
    {
        "id": "hv_010",
        "question": "Sinh viên muốn chuyển ngành học trong trường cần đáp ứng những điều kiện gì?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 14, "article": "Điều 16"}],
        "verified": False,
    },
    {
        "id": "hv_011",
        "question": "Sinh viên năm thứ mấy thì không được phép chuyển ngành học nữa?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 14, "article": "Điều 16"}],
        "verified": False,
    },
    {
        "id": "hv_012",
        "question": "Sinh viên đang bị cảnh báo học vụ có được phép xin chuyển ngành học không?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 14, "article": "Điều 16"}],
        "verified": False,
    },
    {
        "id": "hv_013",
        "question": "Điều kiện để sinh viên được chuyển từ trường khác đến UIT học tập?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 14, "article": "Điều 16"}],
        "verified": False,
    },
    {
        "id": "hv_014",
        "question": "Thủ tục xin thôi học tự nguyện của sinh viên gồm những bước nào?",
        "group": "học vụ",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "hv_015",
        "question": "Học cùng lúc hai chương trình (song ngành) cần thỏa mãn điều kiện gì?",
        "group": "học vụ",
        "relevant": [{"source_file": "1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf", "page": 4, "article": "Điều 6"}],
        "verified": False,
    },
    {
        "id": "hv_016",
        "question": "Sinh viên học song ngành bị xếp loại học lực yếu ở ngành 2 thì bị xử lý ra sao?",
        "group": "học vụ",
        "relevant": [{"source_file": "1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf", "page": 5, "article": "Điều 8"}],
        "verified": False,
    },
    {
        "id": "hv_017",
        "question": "học phí song ngành đại học quốc gia tính như thế nào",
        "group": "học vụ",
        "relevant": [{"source_file": "1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf", "page": 5, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "hv_018",
        "question": "điều kiện xét tuyển liên thông từ đại học lên thạc sĩ BS-MS là gì",
        "group": "học vụ",
        "relevant": [{"source_file": "650-qd-dhqg_3-6-2019-quy_dinh_dao_tao_lien_thong_tu_trinh_do_dai_hoc_len_trinh_do_thac_si.pdf", "page": 6, "article": "Điều 7"}],
        "verified": False,
    },
    {
        "id": "hv_019",
        "question": "Thời gian tối đa để hoàn thành chương trình liên thông BS-MS là bao lâu?",
        "group": "học vụ",
        "relevant": [{"source_file": "650-qd-dhqg_3-6-2019-quy_dinh_dao_tao_lien_thong_tu_trinh_do_dai_hoc_len_trinh_do_thac_si.pdf", "page": 4, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "hv_020",
        "question": "Người học liên thông BS-MS được cấp văn bằng tốt nghiệp như thế nào?",
        "group": "học vụ",
        "relevant": [{"source_file": "650-qd-dhqg_3-6-2019-quy_dinh_dao_tao_lien_thong_tu_trinh_do_dai_hoc_len_trinh_do_thac_si.pdf", "page": 7, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "hv_021",
        "question": "Quy trình nộp đơn xin cấp giấy xác nhận sinh viên tại UIT thực hiện qua kênh nào?",
        "group": "học vụ",
        "relevant": [{"source_file": "28_qd-dhcntt_10-01-2023_quy_dinh_cong_tac_hoc_vu.pdf", "page": 4, "article": None}],
        "verified": False,
    },
    {
        "id": "hv_022",
        "question": "Thời gian giải quyết các đơn từ học vụ thông thường của sinh viên là mấy ngày làm việc?",
        "group": "học vụ",
        "relevant": [{"source_file": "28_qd-dhcntt_10-01-2023_quy_dinh_cong_tac_hoc_vu.pdf", "page": 5, "article": None}],
        "verified": False,
    },

    # =========================================================================
    # NHÓM 3: THI - ĐIỂM (22 câu)
    # =========================================================================
    {
        "id": "td_001",
        "question": "Thang điểm đánh giá học phần tại UIT gồm những thang điểm nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 20"}],
        "verified": False,
    },
    {
        "id": "td_002",
        "question": "Quy đổi từ thang điểm 10 sang thang điểm chữ (A, B, C, D, F) được quy định như thế nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 18, "article": "Điều 21"}],
        "verified": False,
    },
    {
        "id": "td_003",
        "question": "Mức điểm nào trên thang điểm 10 thì bị tính là điểm F (không đạt)?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 18, "article": "Điều 21"}],
        "verified": False,
    },
    {
        "id": "td_004",
        "question": "Điểm chữ C tương ứng với bao nhiêu điểm trên thang 4 và thang 10?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 18, "article": "Điều 21"}],
        "verified": False,
    },
    {
        "id": "td_005",
        "question": "Điểm I là gì và áp dụng trong những trường hợp nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_006",
        "question": "Thời hạn tối đa để sinh viên hoàn thành học phần đang nhận điểm I là bao lâu?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_007",
        "question": "Nếu quá thời hạn giải quyết điểm I mà sinh viên không thi hoàn tất thì điểm I chuyển thành điểm gì?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_008",
        "question": "em bị ốm phải nằm viện không đi thi được thì phải làm gì để được hoãn thi",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_009",
        "question": "Sinh viên vắng thi kết thúc học phần không có lý do chính đáng sẽ nhận điểm mấy?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_010",
        "question": "Vắng thi giữa kỳ vì lý do chính đáng thì điểm giữa kỳ được tính thế nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_011",
        "question": "Sinh viên đi thi muộn bao nhiêu phút thì không được phép vào phòng thi?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 9, "article": "Điều 13"}],
        "verified": False,
    },
    {
        "id": "td_012",
        "question": "Khi đi thi sinh viên bắt buộc phải xuất trình giấy tờ tùy thân nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 8, "article": "Điều 13"}],
        "verified": False,
    },
    {
        "id": "td_013",
        "question": "Những vật dụng nào sinh viên bị cấm tuyệt đối mang vào phòng thi?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 9, "article": "Điều 13"}],
        "verified": False,
    },
    {
        "id": "td_014",
        "question": "Sinh viên mang điện thoại vào phòng thi dù tắt nguồn thì bị xử lý kỷ luật thế nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 11, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "td_015",
        "question": "Sinh viên nhờ người khác thi hộ hoặc đi thi hộ người khác thì bị kỷ luật gì?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 12, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "td_016",
        "question": "Thời hạn sinh viên được nộp đơn phúc khảo bài thi kết thúc học phần là bao lâu?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 10, "article": "Điều 15"}],
        "verified": False,
    },
    {
        "id": "td_017",
        "question": "Sau bao nhiêu ngày kể từ khi hết hạn nộp đơn phúc khảo thì trường công bố điểm phúc khảo?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 10, "article": "Điều 15"}],
        "verified": False,
    },
    {
        "id": "td_018",
        "question": "Điểm M trong bảng điểm là điểm gì và dùng khi nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "td_019",
        "question": "Yêu cầu kỹ thuật đối với khu vực thi trực tuyến của sinh viên là gì?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "809-qd-dhqg_3-7-2023_quy_dinh_dgkqht_theo_hinh_thuc_tt.pdf", "page": 3, "article": "Điều 5"}],
        "verified": False,
    },
    {
        "id": "td_020",
        "question": "Thi trực tuyến bị rớt mạng hoặc mất kết nối thì sinh viên phải xử lý thế nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "809-qd-dhqg_3-7-2023_quy_dinh_dgkqht_theo_hinh_thuc_tt.pdf", "page": 6, "article": "Điều 12"}],
        "verified": False,
    },
    {
        "id": "td_021",
        "question": "Điểm đánh giá bộ phận của một học phần gồm những cột điểm nào?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 20"}],
        "verified": False,
    },
    {
        "id": "td_022",
        "question": "Trọng số điểm thi kết thúc học phần tối thiểu là bao nhiêu phần trăm?",
        "group": "thi-điểm",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 20"}],
        "verified": False,
    },

    # =========================================================================
    # NHÓM 4: TỐT NGHIỆP & CHUẨN ĐẦU RA (21 câu)
    # =========================================================================
    {
        "id": "tn_001",
        "question": "Điều kiện để sinh viên được công nhận tốt nghiệp đại học tại UIT gồm những gì?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 22, "article": "Điều 27"}],
        "verified": False,
    },
    {
        "id": "tn_002",
        "question": "Điểm trung bình tích lũy toàn khóa tối thiểu là bao nhiêu để được xét tốt nghiệp?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 22, "article": "Điều 27"}],
        "verified": False,
    },
    {
        "id": "tn_003",
        "question": "Sinh viên đang bị truy cứu trách nhiệm hình sự có được xét tốt nghiệp không?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 22, "article": "Điều 27"}],
        "verified": False,
    },
    {
        "id": "tn_004",
        "question": "Các hạng xếp loại tốt nghiệp đại học gồm những mức nào?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_005",
        "question": "Điểm GPA tích lũy bao nhiêu thì tốt nghiệp loại Xuất sắc?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_006",
        "question": "Điểm GPA tích lũy bao nhiêu thì tốt nghiệp loại Giỏi?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_007",
        "question": "Những trường hợp nào sinh viên tốt nghiệp loại Giỏi, Xuất sắc bị hạ xuống một bậc?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_008",
        "question": "Thi lại hoặc học lại vượt quá bao nhiêu phần trăm tín chỉ thì bị hạ bậc tốt nghiệp?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_009",
        "question": "Bị kỷ luật từ mức nào trong thời gian học thì bị hạ bậc xếp loại tốt nghiệp?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "tn_010",
        "question": "Chuẩn trình độ ngoại ngữ đầu ra đối với sinh viên tốt nghiệp đại học tối thiểu là bậc mấy?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 4, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "tn_011",
        "question": "Chứng chỉ tiếng Anh nào được công nhận đạt chuẩn tốt nghiệp bậc 3/6 của ĐHQG-HCM?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 4, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "tn_012",
        "question": "Thời hạn công nhận hiệu lực của chứng chỉ ngoại ngữ là bao lâu tính từ ngày cấp?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 5, "article": "Điều 5"}],
        "verified": False,
    },
    {
        "id": "tn_013",
        "question": "Chứng chỉ tiếng Nhật trình độ nào tương đương chuẩn ngoại ngữ bậc 3/6 ĐHQG-HCM?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 5, "article": "Điều 5"}],
        "verified": False,
    },
    {
        "id": "tn_014",
        "question": "Thời hạn cấp bằng tốt nghiệp đại học chính thức là bao nhiêu ngày làm việc kể từ khi có quyết định tốt nghiệp?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 12, "article": "Điều 21"}],
        "verified": False,
    },
    {
        "id": "tn_015",
        "question": "Văn bằng tốt nghiệp đại học được cấp mấy bản chính thức cho mỗi sinh viên?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 3, "article": "Điều 2"}],
        "verified": False,
    },
    {
        "id": "tn_016",
        "question": "Bị mất bằng tốt nghiệp đại học bản chính thì có được cấp lại bản chính không?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 4, "article": "Điều 2"}],
        "verified": False,
    },
    {
        "id": "tn_017",
        "question": "Thủ tục xin cấp bản sao văn bằng chứng chỉ từ sổ gốc gồm những giấy tờ gì?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 17, "article": "Điều 33"}],
        "verified": False,
    },
    {
        "id": "tn_018",
        "question": "Bản sao văn bằng được cấp từ sổ gốc có giá trị pháp lý tương đương bản chính không?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 17, "article": "Điều 33"}],
        "verified": False,
    },
    {
        "id": "tn_019",
        "question": "Hồ sơ đề nghị chỉnh sửa nội dung thông tin trên bằng tốt nghiệp gồm những gì?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 14, "article": "Điều 27"}],
        "verified": False,
    },
    {
        "id": "tn_020",
        "question": "Những trường hợp nào bằng tốt nghiệp đại học bị nhà trường thu hồi hoặc hủy bỏ?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 15, "article": "Điều 29"}],
        "verified": False,
    },
    {
        "id": "tn_021",
        "question": "Sinh viên chưa tích lũy đủ chứng chỉ giáo dục quốc phòng an ninh có được cấp bằng tốt nghiệp đại học không?",
        "group": "tốt nghiệp",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 22, "article": "Điều 27"}],
        "verified": False,
    },

    # =========================================================================
    # NHÓM 5: CÂU KHÓ & KHẨU NGỮ TỰ NHIÊN (20 câu)
    # =========================================================================
    {
        "id": "ck_001",
        "question": "em bị cảnh báo học vụ 2 lần liên tiếp rồi thì kỳ này em có bị đuổi học luôn không bot",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 19"}],
        "verified": False,
    },
    {
        "id": "ck_002",
        "question": "vừa học lại môn F vừa học cải thiện môn C thì điểm tích lũy GPA toàn khóa tính toán như thế nào",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 9, "article": "Điều 10"}],
        "verified": False,
    },
    {
        "id": "ck_003",
        "question": "em đang học năm 2 ngành kỹ thuật máy tính muốn xin học song bằng sang ngành khoa học máy tính thì cần chuẩn bị hồ sơ gì và nộp ở đâu",
        "group": "câu khó",
        "relevant": [{"source_file": "1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf", "page": 4, "article": "Điều 7"}],
        "verified": False,
    },
    {
        "id": "ck_004",
        "question": "bị điểm F môn thể dục và giáo dục quốc phòng thì có bị tính vào điểm trung bình tích lũy học kỳ để xét học bổng hay cảnh báo học vụ không",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 18, "article": "Điều 21"}],
        "verified": False,
    },
    {
        "id": "ck_005",
        "question": "điểm I để quá 2 học kỳ chính không chịu thi trả nợ thì chuyện gì xảy ra với môn học đó",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 19, "article": "Điều 22"}],
        "verified": False,
    },
    {
        "id": "ck_006",
        "question": "em muốn bảo lưu để đi du học 1 năm thì cần nộp những giấy tờ gì và trường có giới hạn số lần bảo lưu không",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 15, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "ck_007",
        "question": "điểm GPA của em là 3.65 nhưng bị học lại 6 tín chỉ và bị khiển trách 1 lần thì tốt nghiệp được loại xuất sắc hay bị hạ bằng giỏi",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 23, "article": "Điều 28"}],
        "verified": False,
    },
    {
        "id": "ck_008",
        "question": "sinh viên năm 3 có điểm trung bình 6.8 muốn đăng ký chương trình tích hợp BS-MS lên thạc sĩ được không",
        "group": "câu khó",
        "relevant": [{"source_file": "650-qd-dhqg_3-6-2019-quy_dinh_dao_tao_lien_thong_tu_trinh_do_dai_hoc_len_trinh_do_thac_si.pdf", "page": 6, "article": "Điều 7"}],
        "verified": False,
    },
    {
        "id": "ck_009",
        "question": "đang trong phòng thi mà bị xỉu hay đau bụng dữ dội phải đi cấp cứu thì điểm bài thi hôm đó xử lý ra sao",
        "group": "câu khó",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 9, "article": "Điều 13"}],
        "verified": False,
    },
    {
        "id": "ck_010",
        "question": "bằng tốt nghiệp của trường mình ghi xếp loại bằng tiếng Việt hay cả tiếng Anh và có ghi hệ chính quy không",
        "group": "câu khó",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 5, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "ck_011",
        "question": "em nộp chứng chỉ tiếng Anh TOEIC 4 kỹ năng hay 2 kỹ năng mới được trường chấp nhận chuẩn đầu ra",
        "group": "câu khó",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 4, "article": "Điều 4"}],
        "verified": False,
    },
    {
        "id": "ck_012",
        "question": "chứng chỉ ngoại ngữ thi từ năm nhất đến năm tư tốt nghiệp còn hạn sử dụng để xét chuẩn đầu ra không",
        "group": "câu khó",
        "relevant": [{"source_file": "170_qd_dhqg_27_02_2018_cong_nhan_chung_chi_ngoai_ngu.pdf", "page": 5, "article": "Điều 5"}],
        "verified": False,
    },
    {
        "id": "ck_013",
        "question": "rớt môn giáo dục thể chất có bị cảnh báo học vụ và có được xét tốt nghiệp đúng hạn không",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 22, "article": "Điều 27"}],
        "verified": False,
    },
    {
        "id": "ck_014",
        "question": "sinh viên bị kỷ luật đình chỉ thi 1 môn thì môn đó bị điểm mấy và có bị đình chỉ các môn thi tiếp theo không",
        "group": "câu khó",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 11, "article": "Điều 17"}],
        "verified": False,
    },
    {
        "id": "ck_015",
        "question": "học cùng lúc 2 ngành thì môn trùng nhau ở ngành 1 có được chuyển điểm tự động sang ngành 2 không",
        "group": "câu khó",
        "relevant": [{"source_file": "1195-qd-dhqg_27-9-2019_quy_dinh_dao_tao_song_nganh_dhqg.pdf", "page": 5, "article": "Điều 9"}],
        "verified": False,
    },
    {
        "id": "ck_016",
        "question": "bị mất thẻ sinh viên trước giờ thi 15 phút thì có được vào phòng thi làm bài không và cần làm gì",
        "group": "câu khó",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 8, "article": "Điều 13"}],
        "verified": False,
    },
    {
        "id": "ck_017",
        "question": "hết thời gian đào tạo tối đa 6 năm mà chưa trả hết nợ môn thì trường xử lý thế nào",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 17, "article": "Điều 19"}],
        "verified": False,
    },
    {
        "id": "ck_018",
        "question": "nếu điểm phúc khảo lệch so với điểm ban đầu thì khi nào được sửa điểm chính thức vào hệ thống",
        "group": "câu khó",
        "relevant": [{"source_file": "1139_qd-dhcntt_20-12-2022_to_chuc_thi_cac_mon_hoc_he_dai_hoc_chinh_quy.pdf", "page": 10, "article": "Điều 15"}],
        "verified": False,
    },
    {
        "id": "ck_019",
        "question": "chứng chỉ tin học MOS hay IC3 có được công nhận chuẩn đầu ra tin học ở trường không",
        "group": "câu khó",
        "relevant": [{"source_file": "1372-qd-dhqg_10-10-2022_quy_che_van_bang_chung_chi.pdf", "page": 4, "article": "Điều 2"}],
        "verified": False,
    },
    {
        "id": "ck_020",
        "question": "sinh viên chuyển trường từ Đại học Bách Khoa sang UIT thì các môn đã học bên Bách Khoa có được bảo lưu chuyển điểm không",
        "group": "câu khó",
        "relevant": [{"source_file": "quy_che_dao_tao.pdf", "page": 14, "article": "Điều 16"}],
        "verified": False,
    },
]


def draft_questions() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    count = len(DRAFT_ITEMS)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in DRAFT_ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo thành công {count} câu hỏi nháp tại: {OUTPUT_FILE}")
    print("Các nhóm câu hỏi gồm:")
    groups = {}
    for item in DRAFT_ITEMS:
        g = item["group"]
        groups[g] = groups.get(g, 0) + 1
    for g, c in sorted(groups.items()):
        print(f"  - {g}: {c} câu")


if __name__ == "__main__":
    draft_questions()
