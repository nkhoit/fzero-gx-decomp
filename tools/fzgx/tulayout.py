"""Retail TU section layout for one function, as C definitions a body can be written against.

A retail function that addresses its translation unit's objects off compiler-made section bases
(`addi r30, r6, lbl_1_data_35AC8@l` followed by `lwz r4, 0x348(r6)`, `addi r5, r30, 0x7320`, or a
`lis/addi/lwz 0(rX)` of a .bss object) can only be reproduced when the unit defines those objects
itself: MWCC then addresses every object of a section off one base register, and only an object
referenced at its own offset 0 is base-relative. An extern struct over the cluster never gives
that shape (see CLAUDE.md, "Retail TU section layout").

`layout()` reads the function's retail assembly, follows the base registers, and returns one
object per referenced address for each section the function touches:

  .data / .rodata  retail bytes (the oracle compares them), cut at every referenced word: MWCC
                   puts each initialized aggregate on a word boundary and every initialized
                   scalar after all aggregates, so only u8 arrays at word addresses keep the
                   retail offsets; an unaligned reference stays interior to the object before
  .bss             one object per dtk symbol in the referenced span, split again where the
                   function reaches past the symbol its base was built from

plus a `.fzgxpool` primer per section that fixes the emission order. The names follow the
binding rules of `poolfix`/the oracle (`fzgx_pool_<base>[_OFF]`, `fzgx_obj_<sym>`,
`<sym>__fzgx_offset_N`, `fzgx_pool_gap_bss_<addr>`), so a body that matches with them submits
as a full match and links. The table of references tells the body author which object and
offset each retail access uses.

A reference cuts only where the function leaves the extent of the symbol its base register was
built from (the register is then the section's base, not a pointer to one object): interior
offsets of that symbol are fields, and cutting there would only change the C spelling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .evidence import retail_bytes

# Definition order decides which base takes which callee-saved register; .bss first reproduces
# the matched memcard.c bodies (data first swapped r29/r30 in fn_1_C062C and fn_1_C0E00).
SECTIONS = ('.bss', '.data', '.rodata')
VOLATILE = {0} | set(range(3, 13))
WIDTH = {'lbz': 1, 'lbzu': 1, 'stb': 1, 'stbu': 1, 'lhz': 2, 'lha': 2, 'lhzu': 2, 'lhau': 2,
         'sth': 2, 'sthu': 2, 'lwz': 4, 'lwzu': 4, 'stw': 4, 'stwu': 4, 'lfs': 4, 'lfsu': 4,
         'stfs': 4, 'stfsu': 4, 'lfd': 8, 'lfdu': 8, 'stfd': 8, 'stfdu': 8, 'psq_l': 8, 'psq_st': 8,
         'lmw': 4, 'stmw': 4}
CTYPE = {'lbz': 'u8', 'stb': 'u8', 'lhz': 'u16', 'lha': 's16', 'sth': 'u16', 'lwz': 'u32',
         'stw': 'u32', 'lfs': 'f32', 'stfs': 'f32', 'lfd': 'f64', 'stfd': 'f64'}
_LINE = re.compile(r'^[0-9A-Fa-f]{8}:\s+(\S+)\s*(.*)$')
_MEM = re.compile(r'^(r\d+|f\d+),\s*(-?(?:0x[0-9A-Fa-f]+|\d+)|(\w+)@l)\((r\d+)\)$')


@dataclass
class Ref:
    section: str
    addr: int
    op: str               # mnemonic, or 'addr' for a materialized address
    at: int               # instruction address
    root: str = ''        # the symbol whose address the base register was built from


@dataclass
class Piece:
    name: str
    addr: int
    size: int
    scalar: bool          # a u8/u16/u32 scalar rather than an array
    ctype: str = 'u8'
    refs: List[Ref] = field(default_factory=list)


def _reg(text: str) -> Optional[int]:
    m = re.fullmatch(r'r(\d+)', text.strip())
    return int(m.group(1)) if m else None


def references(p, fn) -> List[Ref]:
    """Every same-module data/rodata/bss address the function loads, stores or materializes.

    A walk over the retail listing: `lis rX, S@ha` / `addi rY, rX, S@l` start an address,
    `addi`/`mr` carry it, a displacement access through it is a reference. Any other write kills
    the register, a call kills the volatile ones, and a label keeps only what every forward
    branch into it agrees on (a pointer chosen on two paths is a variable, not a base). A label
    reached only by a backward branch or a jump table keeps the callee-saved registers."""
    syms = p.symbols(fn.symbol.module)
    out: List[Ref] = []
    state: Optional[dict] = {}
    pending: Dict[str, dict] = {}
    last: dict = {}

    def note(st, sym: str, off: int, op: str, at: int):
        s = syms.get(sym)
        if s is not None and s.section in SECTIONS and s.kind == 'object':
            out.append(Ref(s.section, s.addr + off, op, at, sym))

    def meet(x, y):
        if x is None:
            return dict(y) if y is not None else None
        if y is None:
            return dict(x)
        return {k: v for k, v in x.items() if y.get(k) == v}

    for line in fn.asm:
        label = line.strip()
        if label.endswith(':') and not _LINE.match(line):
            merged = meet(state, pending.pop(label[:-1], None))
            if merged is None:
                merged = {k: v for k, v in last.items() if k[1] >= 13}
            state = merged
            continue
        m = _LINE.match(line)
        if not m or state is None:
            continue
        at = int(line.split(':', 1)[0], 16)
        op, args = m.group(1).rstrip('.+-'), m.group(2)
        parts = [x.strip() for x in args.split(',')] if args else []
        dest = _reg(parts[0]) if parts else None
        hi = {k[1]: v for k, v in state.items() if k[0] == 'hi'}
        addr = {k[1]: v for k, v in state.items() if k[0] == 'addr'}

        def kill(r):
            state.pop(('hi', r), None)
            state.pop(('addr', r), None)

        if op.startswith('b') and op not in ('bl', 'bctrl', 'blrl'):
            target = re.search(r'(\.L_[0-9A-Fa-f]+)$', args or '')
            if target:
                pending[target.group(1)] = meet(pending.get(target.group(1)), state)
            if op in ('b', 'blr', 'bctr'):
                last, state = dict(state), None
            continue
        if op == 'lis' and len(parts) == 2 and parts[1].endswith('@ha'):
            kill(dest)
            state[('hi', dest)] = parts[1][:-3]
            continue
        if op in ('addis', 'subi') and len(parts) == 3 and _reg(parts[1]) in addr \
                and re.fullmatch(r'-?(0x[0-9A-Fa-f]+|\d+)', parts[2]):
            base, off = addr[_reg(parts[1])]
            imm = int(parts[2], 0)
            off += (imm << 16) if op == 'addis' else -imm
            kill(dest)
            state[('addr', dest)] = (base, off)
            if op == 'subi':
                note(state, base, off, 'addr', at)
            continue
        if op == 'addi' and len(parts) == 3:
            src = _reg(parts[1])
            if parts[2].endswith('@l') and hi.get(src) == parts[2][:-2]:
                kill(dest)
                state[('addr', dest)] = (parts[2][:-2], 0)
                note(state, parts[2][:-2], 0, 'addr', at)
                continue
            if src in addr and src != dest and re.fullmatch(r'-?(0x[0-9A-Fa-f]+|\d+)', parts[2]):
                base, off = addr[src]
                kill(dest)
                state[('addr', dest)] = (base, off + int(parts[2], 0))
                note(state, base, off + int(parts[2], 0), 'addr', at)
                continue
        if op == 'mr' and len(parts) == 2 and _reg(parts[1]) in addr:
            kill(dest)
            state[('addr', dest)] = addr[_reg(parts[1])]
            continue
        base_op = op[:-1] if op.endswith('u') and op[:-1] in WIDTH else op
        if base_op in WIDTH and len(parts) >= 2:
            mm = _MEM.match(', '.join(parts[:2]))
            if mm:
                rb = _reg(mm.group(4))
                if mm.group(3) and hi.get(rb) == mm.group(3):
                    note(state, mm.group(3), 0, base_op, at)
                elif rb in addr and not mm.group(3):
                    base, off = addr[rb]
                    note(state, base, off + int(mm.group(2), 0), base_op, at)
                if op != base_op:
                    kill(rb)  # update forms move the base
            if op.startswith(('l', 'psq_l')) and dest is not None and parts[0].startswith('r'):
                kill(dest)
            continue
        if op in ('bl', 'bctrl', 'blrl'):
            for r in VOLATILE:
                kill(r)
            continue
        if dest is not None and not op.startswith(('cmp', 'st', 'mt', 'tw', 'dcb', 'icb')):
            kill(dest)
    return out


def _containing(syms_by_addr, addr):
    best = None
    for s in syms_by_addr:
        if s.addr <= addr:
            best = s
        else:
            break
    return best


DATA_TYPE = {'byte': 'u8', '2byte': 'u16', '4byte': 'u32', 'float': 'f32', 'double': 'f64',
             'string': 'char'}
SIZE = {'u8': 1, 'char': 1, 'u16': 2, 's16': 2, 'u32': 4, 'f32': 4, 'f64': 8}


def _element(ordered, piece: Piece) -> str:
    """Element type of a .bss array: dtk's data kind at a symbol start, else the access at the
    piece's first byte; u8 when the size is no multiple of it."""
    owner = _containing(ordered, piece.addr)
    t = None
    if owner is not None and owner.addr == piece.addr:
        t = DATA_TYPE.get((owner.attrs or {}).get('data', ''))
    if t is None:
        first = [r.op for r in piece.refs if r.addr == piece.addr and r.op in CTYPE]
        t = CTYPE[first[0]] if first else 'u8'
    if t == 'f64' or piece.size % SIZE[t] or piece.addr % SIZE[t]:
        return 'u8'
    return t


def _bss_name(ordered, addr: int) -> str:
    """The names the oracle binds to retail .bss: `fzgx_obj_<sym>` at a dtk symbol,
    `<sym>__fzgx_offset_N` inside one, `fzgx_pool_gap_bss_<addr>` past its size. Any other
    name leaves the private section in the object and the function ends pool-only."""
    owner = _containing(ordered, addr)
    if owner and owner.addr == addr:
        return f'fzgx_obj_{owner.name}'
    if owner and addr < owner.addr + owner.size:
        return f'{owner.name}__fzgx_offset_{addr - owner.addr:X}'
    return f'fzgx_pool_gap_bss_{addr:X}'


def _scalar_type(addr: int, size: int) -> str:
    if size == 4 and addr % 4 == 0:
        return 'u32'
    if size == 2 and addr % 2 == 0:
        return 'u16'
    return 'u8'


def _split(name: str, start: int, end: int, extra_name) -> List[Piece]:
    """[start, end) as objects MWCC places contiguously: u8 scalars up to the next word, then one
    4-aligned array (or a scalar when the remainder is a single u8/u16/u32)."""
    pieces = []
    a = start
    while a < end and a % 4:
        pieces.append(Piece(name if a == start else extra_name(a), a, 1, True))
        a += 1
    if a < end:
        n = name if a == start else extra_name(a)
        if end - a in (1, 2, 4):
            pieces.append(Piece(n, a, end - a, True, _scalar_type(a, end - a)))
        else:
            pieces.append(Piece(n, a, end - a, False))
    return pieces


def layout(p, symbol: str) -> Dict[str, object]:
    sym = p.resolve(symbol)
    fn = p.function_asm(sym.module).get(sym.name)
    if fn is None:
        return dict(ok=False, symbol=symbol, notes=['no retail assembly for this function'])
    syms = p.symbols(sym.module)
    refs = references(p, fn)
    result = dict(ok=True, symbol=sym.name, module=sym.module, sections={}, notes=[])
    for section in SECTIONS:
        srefs = [r for r in refs if r.section == section]
        if not srefs:
            continue
        ordered = sorted((s for s in syms.values() if s.section == section and s.kind == 'object'),
                         key=lambda s: s.addr)
        start = min(r.addr for r in srefs)
        first = _containing(ordered, start)
        if first is None:
            result['notes'].append(f'{section}: no dtk symbol at or below {start:#x}')
            continue
        start = first.addr
        end = start
        for r in srefs:
            owner = _containing(ordered, r.addr)
            width = WIDTH.get(r.op, 1)
            end = max(end, r.addr + width, owner.addr + owner.size if owner else 0)
        # An object starts at every dtk symbol a reference names and at every address the
        # function reaches beyond the extent of the symbol its base register was built from
        # (the base is then the section's, not an object pointer). Interior offsets of the
        # root symbol are fields: cutting there would change the C spelling, not the layout.
        cuts = {start, end}
        for r in srefs:
            root = syms.get(r.root)
            if r.addr == root.addr or not (root.addr <= r.addr < root.addr + max(root.size, 1)):
                cuts.add(r.addr)
            if r.addr == root.addr and root.size:
                cuts.add(root.addr + root.size)
        if section == '.bss':
            cuts |= {s.addr for s in ordered if start <= s.addr < end}
            cuts |= {s.addr + s.size for s in ordered if start < s.addr + s.size < end}
        cuts = sorted(c for c in cuts if start <= c <= end)
        if section != '.bss':
            # MWCC places every initialized aggregate on a word boundary and emits initialized
            # scalars after all aggregates, so an object can only start at a word: an unaligned
            # reference stays an interior address of the object before it.
            odd = [c for c in cuts[1:-1] if c % 4]
            if odd:
                result['notes'].append(f'{section}: unaligned references kept interior: '
                                       + ', '.join(f'+{c - start:#x}' for c in odd))
            cuts = [c for c in cuts if c % 4 == 0 or c in (start, end)]
        pieces: List[Piece] = []
        for a, z in zip(cuts, cuts[1:]):
            if section == '.bss':
                name = _bss_name(ordered, a)
                extra = lambda x: _bss_name(ordered, x)
            else:
                name = f'fzgx_pool_{first.name}' + ('' if a == start else f'_{a - start:X}')
                extra = lambda x: f'fzgx_pool_{first.name}_{x - start:X}'
            pieces.extend(_split(name, a, z, extra) if section == '.bss' else [Piece(name, a, z - a, False)])
        for piece in pieces:
            piece.refs = [r for r in srefs if piece.addr <= r.addr < piece.addr + piece.size]
            if section == '.bss' and not piece.scalar:
                piece.ctype = _element(ordered, piece)
        data = None
        if section != '.bss':
            data = retail_bytes(p, sym.module, first.name, 0, end - start)
            if data is None:
                result['notes'].append(f'{section}: retail bytes of {first.name}+[0, {end - start:#x}) unavailable')
                continue
        result['sections'][section] = dict(base=first.name, start=start, end=end, pieces=pieces, data=data)
    return result


def _bytes(b: bytes, indent='    ') -> str:
    rows = [indent + ', '.join(f'0x{x:02X}' for x in b[i:i + 16]) + ',' for i in range(0, len(b), 16)]
    return '\n'.join(rows)


def access(piece: Piece, ref: Ref) -> str:
    off = ref.addr - piece.addr
    if ref.op == 'addr':
        return f'{piece.name}' + (f' + 0x{off:X}' if off else '') if not piece.scalar else f'&{piece.name}'
    t = CTYPE.get(ref.op, f'/* {ref.op} */ u32')
    ptr = f'{piece.name}' if not piece.scalar else f'&{piece.name}'
    return f'*({t} *)((u8 *){ptr} + 0x{off:X})' if off else f'*({t} *){ptr}'


def render(r: Dict[str, object]) -> str:
    if not r['ok']:
        return '\n'.join(r['notes'])
    out = [f'/* retail TU section layout for {r["symbol"]} (fzgx tulayout): MWCC addresses these off'
           ' section bases only when the unit defines them */']
    for section, s in r['sections'].items():
        out.append(f'\n/* {section}: {s["base"]} +[0, 0x{s["end"] - s["start"]:X}) */')
        for piece in s['pieces']:
            if section == '.bss':
                n = piece.size // SIZE.get(piece.ctype, 1)
                out.append(f'{piece.ctype} {piece.name};' if piece.scalar else f'{piece.ctype} {piece.name}[0x{n:X}];')
                continue
            qual = 'static const ' if section == '.rodata' else 'static '
            b = s['data'][piece.addr - s['start']:piece.addr - s['start'] + piece.size]
            note = f'  /* fzgx-allow: A1 retail {section[1:]} bytes: {s["base"]} +0x{piece.addr - s["start"]:X} */'
            if piece.scalar:
                value = int.from_bytes(b, 'big')
                out.append(f'{qual}{piece.ctype} {piece.name} = 0x{value:0{2 * piece.size}X};{note}')
            else:
                out.append(f'{qual}u8 {piece.name}[0x{piece.size:X}] = {{{note}\n{_bytes(b)}\n}};')
        out.append('#pragma section code_type ".fzgxpool"')
        out.append(f'static void fzgx_{section[1:]}_layout(void) {{')
        out.append(f'    volatile u8 s;  /* fzgx-allow: S2 layout primer sink: MWCC emits {section} objects in first-access order */')
        for piece in s['pieces']:
            out.append(f'    s = *(u8 *)&{piece.name};')
        out.append('}')
        out.append('#pragma section code_type ".text"')
    out.append('\n/* references (retail instruction -> C spelling):')
    for section, s in r['sections'].items():
        for piece in s['pieces']:
            for ref in piece.refs:
                out.append(f'   {ref.at:08X} {ref.op:<5} {section:<7} +0x{ref.addr - s["start"]:<5X} {access(piece, ref)}')
    out.append('*/')
    if r['notes']:
        out.append('/* notes: ' + '; '.join(r['notes']) + ' */')
    return '\n'.join(out) + '\n'
