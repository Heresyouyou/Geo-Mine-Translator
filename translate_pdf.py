"""
PDF 翻译 CLI — 选定 PDF → 翻译 → 同目录输出 【已翻译】xxx.pdf

用法:
    python translate_pdf.py D:\\论文\\test.pdf
    python translate_pdf.py "D:\\多文件\\*.pdf"       (通配符批量)
    python translate_pdf.py --list D:\\论文\\*.pdf   (批量)
"""
import sys, time, os, glob, argparse
from pathlib import Path

_THIS = Path(__file__).parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))


def translate_one(pdf_path: str) -> str:
    """翻译单个 PDF, 输出同目录 【已翻译】xxx.pdf, 返回输出路径

    使用 BabelDOC (版式保留, 公式占位符) + HY-MT1.5 翻译引擎。
    """
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'文件不存在: {pdf_path}')

    # 输出路径: 同目录 / 【已翻译】xxx.pdf
    p = Path(pdf_path)
    out = p.parent / f'【已翻译】{p.name}'

    if out.exists():
        print(f'  ⚠️  输出已存在, 跳过: {out.name}')
        return str(out)

    print(f'📄 {p.name} ({p.stat().st_size/1024/1024:.1f}MB)')
    t0 = time.time()

    from run_babeldoc import translate_pdf as babeldoc_translate
    # 输出到 translated/ 目录
    out_dir = p.parent / 'translated'
    out_dir.mkdir(exist_ok=True)
    mono, dual = babeldoc_translate(pdf_path, str(out_dir))
    # BabelDOC 输出 *-mono.pdf / *-dual.pdf, 复制 mono 到 【已翻译】xxx.pdf
    if mono and os.path.exists(mono):
        import shutil
        shutil.copy2(mono, str(out))
        elapsed = time.time() - t0
        print(f'  ✅ BabelDOC {elapsed:.1f}s → {out.name}')
        if dual and os.path.exists(dual):
            print(f'     双语版: {Path(dual).name}')
    else:
        raise RuntimeError('BabelDOC 未产出 PDF')
    return str(out)


def expand_pattern(pattern: str) -> list:
    """展开通配符"""
    paths = glob.glob(pattern)
    if paths:
        return paths
    return [pattern]


def main():
    ap = argparse.ArgumentParser(description='PDF 英译中 (本地 HY-MT1.5 + 地学术语库)')
    ap.add_argument('files', nargs='+', help='PDF 文件或通配符')
    ap.add_argument('--list', action='store_true', help='把所有 PDF 列出来不翻译 (dry-run)')
    args = ap.parse_args()

    # 展开所有文件
    all_files = []
    for pat in args.files:
        all_files.extend(expand_pattern(pat))

    # 过滤非 PDF
    all_files = [f for f in all_files if f.lower().endswith('.pdf')]

    if not all_files:
        print('❌ 没有找到 PDF 文件')
        sys.exit(1)

    if args.list:
        print(f'找到 {len(all_files)} 个 PDF:')
        for f in all_files:
            print(f'  📄 {f}')
        return

    print(f'=== 批量翻译 {len(all_files)} 个 PDF ===\n')
    ok, fail = 0, 0
    t0 = time.time()

    for f in all_files:
        try:
            translate_one(f)
            ok += 1
        except Exception as e:
            print(f'  ❌ 失败: {e}')
            fail += 1

    total = time.time() - t0
    print(f'\n=== 完成: {ok} 成功, {fail} 失败, 总耗时 {total:.0f}s ===')


if __name__ == '__main__':
    main()
