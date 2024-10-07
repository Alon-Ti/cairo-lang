import os
import json

from typing import List, Tuple

from starkware.cairo.lang.compiler.instruction import (
    BytecodeData,
    BytecodeElement,
    M31Instruction,
)

OPCODE_LIST_FILE = os.path.join(os.path.dirname(__file__), "opcode_list.json")
OPCODE_DICT = {inst: i for i, inst in enumerate(json.load(open(OPCODE_LIST_FILE)))}
IDX_TO_OPCODE = {i: inst for i, inst in enumerate(json.load(open(OPCODE_LIST_FILE)))}


def encode_instruction(element: BytecodeElement, prime: int) -> List[int]:
    """
    Given an M31Instruction, returns a list of 1 or 2 integers representing the instruction.
    """
    if isinstance(element, BytecodeData):
        return [element.data % prime]
    assert isinstance(element, M31Instruction)
    instruction = [OPCODE_DICT[element.opcode]] + (element.operands + [0] * 4)[:3]
    return [x % prime for x in instruction]

def decode_instruction(encoded: Tuple[int]) -> M31Instruction:
    """
    Given an encoded instruction, returns the decoded instruction.
    """
    opcode = IDX_TO_OPCODE[encoded[0]]
    operands = encoded[1:]
    return M31Instruction(opcode=opcode, operands=operands)

def is_call_instruction(inst: M31Instruction) -> bool:
    return IDX_TO_OPCODE[inst.opcode].startswith("call")
