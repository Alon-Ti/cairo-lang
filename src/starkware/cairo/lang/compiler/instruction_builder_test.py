import pytest

from starkware.cairo.lang.compiler.error_handling import get_location_marks
from starkware.cairo.lang.compiler.instruction import BytecodeElement, M31Instruction, Register
from starkware.cairo.lang.compiler.instruction_builder import (
    InstructionBuilderError,
    build_instruction,
)
from starkware.cairo.lang.compiler.parser import parse_instruction


def parse_and_build(inst: str) -> BytecodeElement:
    """
    Parses the given instruction and builds the BytecodeElement instance.
    """
    return build_instruction(parse_instruction(inst))


def test_assert_eq():
    assert parse_and_build("[ap] = [fp], ap++") == M31Instruction(
        opcode="assert_ap_deref_fp_appp", operands=[0, 0]
    )
    assert parse_and_build("[fp - 3] = [fp + 7]") == M31Instruction(
        opcode="assert_fp_deref_fp", operands=[-3, 7]
    )
    assert parse_and_build("[ap - 3] = [ap]") == M31Instruction(
        opcode="assert_ap_deref_ap", operands=[-3, 0]
    )


def test_assert_eq_reversed():
    assert parse_and_build("5 = [fp + 1]") == parse_and_build("[fp + 1] = 5")
    assert parse_and_build("[[ap + 2] + 3] = [fp + 1], ap++") == parse_and_build(
        "[fp + 1] = [[ap + 2] + 3], ap++"
    )
    assert parse_and_build("[ap] + [fp] = [fp + 1]") == parse_and_build("[fp + 1] = [ap] + [fp]")


def test_assert_eq_instruction_failures():
    verify_exception(
        """\
fp - 3 = [fp]
^****^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
ap = [fp]
^^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
[ap] = [fp * 3]
        ^****^
Expected '+' or '-', found: '*'.
"""
    )
    verify_exception(
        """\
[ap] = [fp + 32768]
             ^***^
Expected a constant offset in the range [-2^15, 2^15).
"""
    )
    verify_exception(
        """\
[ap] = [fp - 32769]
             ^***^
Expected a constant offset in the range [-2^15, 2^15).
"""
    )
    verify_exception(
        """\
[5] = [fp]
 ^
Expected a register. Found: 5.
"""
    )
    verify_exception(
        """\
[x + 7] = [15]
 ^
Expected a register. Found: x.
"""
    )
    # Make sure that if the instruction is invalid, the error is given for its original form,
    # rather than the reversed form.
    verify_exception(
        """\
[[ap + 1]] = [[ap + 1]]
 ^******^
Expected a register. Found: [ap + 1].
"""
    )


def test_assert_eq_double_dereference():
    assert parse_and_build("[ap + 2] = [[fp]]") == M31Instruction(
        opcode="assert_ap_double_deref_fp", operands=[2, 0, 0]
    )
    assert parse_and_build("[ap + 2] = [[ap - 4] + 7], ap++") == M31Instruction(
        opcode="assert_ap_double_deref_ap_appp", operands=[2, -4, 7]
    )


def test_assert_eq_double_dereference_failures():
    verify_exception(
        """\
[ap + 2] = [[fp + 32768] + 17]
                  ^***^
Expected a constant offset in the range [-2^15, 2^15).
"""
    )
    verify_exception(
        """\
[ap + 2] = [[fp * 32768] + 17]
             ^********^
Expected '+' or '-', found: '*'.
"""
    )


def test_assert_eq_imm():
    assert parse_and_build("[ap + 2] = 1234567890") == M31Instruction(
        opcode="assert_ap_imm", operands=[2, 1234567890]
    )


def test_assert_eq_operation():
    assert parse_and_build("[ap + 1] = [ap - 7] * [fp + 3]") == M31Instruction(
        opcode="assert_ap_mul_ap_fp", operands=[1, -7, 3]
    )
    assert parse_and_build("[ap + 10] = [fp] + 1234567890") == M31Instruction(
        opcode="assert_ap_add_imm_fp", operands=[10, 0, 1234567890]
    )
    assert parse_and_build("[fp - 3] = [ap + 7] * [ap + 8]") == M31Instruction(
        opcode="assert_fp_mul_ap_ap", operands=[-3, 7, 8]
    )


def test_inverse_syntactic_sugar():
    assert parse_and_build("[fp] = [ap + 10] - [fp - 1]") == parse_and_build(
        "[ap + 10] = [fp] + [fp - 1]"
    )
    assert parse_and_build("[fp] = [ap + 10] / [fp - 1]") == parse_and_build(
        "[ap + 10] = [fp] * [fp - 1]"
    )


def test_inverse_syntactic_sugar_failures():
    # The syntactic sugar for sub is op0 = dst - op1.
    verify_exception(
        """\
[fp] = [ap + 10] - 1234567890
                   ^********^
Subtraction and division are not supported for immediates.
"""
    )
    verify_exception(
        """\
[fp] = [ap + 10] / 1234567890
                   ^********^
Subtraction and division are not supported for immediates.
"""
    )
    verify_exception(
        """\
1234567890 = [ap + 10] - [fp]
^********^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
[ap] = [[fp]] - [ap]
        ^**^
Expected a register. Found: [fp].
"""
    )
    verify_exception(
        """\
[ap] = 5 - [ap]
       ^
Expected a dereference expression.
"""
    )


def test_assert_eq_operation_failures():
    verify_exception(
        """\
[ap + 1] = 1234 * [fp]
           ^**^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
[ap + 1] = [fp] + [fp] * [fp]
                  ^*********^
Expected a constant expression or a dereference expression.
"""
    )


def test_jump_instruction():
    assert parse_and_build("jmp rel [ap + 1] + [fp - 7]") == M31Instruction(
        opcode="jmp_rel_add_ap_fp", operands=[1, -7]
    )
    assert parse_and_build("jmp abs 123, ap++") == M31Instruction(
        opcode="jmp_abs_imm_appp", operands=[123]
    )
    assert parse_and_build("jmp rel [ap + 1] + [ap - 7]") == M31Instruction(
        opcode="jmp_rel_add_ap_ap", operands=[1, -7]
    )


def test_jnz_instruction():
    assert parse_and_build("jmp rel [fp - 1] if [fp - 7] != 0") == M31Instruction(
        opcode="jnz_fp_fp", operands=[-1, -7]
    )
    assert parse_and_build("jmp rel [ap - 1] if [fp - 7] != 0") == M31Instruction(
        opcode="jnz_ap_fp", operands=[-1, -7]
    )
    assert parse_and_build("jmp rel 123 if [ap] != 0, ap++") == M31Instruction(
        opcode="jnz_imm_ap_appp", operands=[123, 0]
    )


def test_jnz_instruction_failures():
    verify_exception(
        """\
jmp rel [fp] if 5 != 0
                ^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
jmp rel [ap] if [fp] + 3 != 0
                ^******^
Expected a dereference expression.
"""
    )
    verify_exception(
        """\
jmp rel [ap] if [fp * 3] != 0
                 ^****^
Expected '+' or '-', found: '*'.
"""
    )
    verify_exception(
        """\
jmp rel [ap] + [fp] if [fp] != 0
        ^*********^
Invalid expression for jmp offset.
"""
    )


def test_call_instruction():
    assert parse_and_build("call abs [fp + 4]") == M31Instruction(
        opcode="call_abs_fp", operands=[4]
    )

    assert parse_and_build("call rel [fp + 4]") == M31Instruction(
        opcode="call_rel_fp", operands=[4]
    )
    assert parse_and_build("call rel [ap + 4]") == M31Instruction(
        opcode="call_rel_ap", operands=[4]
    )
    assert parse_and_build("call rel 123") == M31Instruction(
        opcode="call_rel_imm", operands=[123]
    )


def test_call_instruction_failures():
    verify_exception(
        """\
call rel [ap] + 5
         ^******^
Invalid offset for call.
"""
    )
    verify_exception(
        """\
call rel 5, ap++
^**************^
ap++ may not be used with the call opcode.
"""
    )


def test_ret_instruction():
    assert parse_and_build("ret") == M31Instruction(
        opcode="ret", operands=[]
    )


def test_ret_instruction_failures():
    verify_exception(
        """\
ret, ap++
^*******^
ap++ may not be used with the ret opcode.
"""
    )


def test_addap_instruction():
    assert parse_and_build("ap += [fp + 4] + [fp]") == M31Instruction(
        opcode="addap_add_fp_fp", operands=[4, 0]
    )
    assert parse_and_build("ap += [ap + 4] + [ap]") == M31Instruction(
        opcode="addap_add_ap_ap", operands=[4, 0]
    )
    assert parse_and_build("ap += 123") == M31Instruction(
        opcode="addap_imm", operands=[123]
    )


def test_addap_instruction_failures():
    verify_exception(
        """\
ap += 5, ap++
^***********^
ap++ may not be used with the addap opcode.
"""
    )


def verify_exception(code_with_err):
    """
    Gets a string with three lines:
        code
        location marks
        error message
    Verifies that parsing the code results in the given error.
    """
    code = code_with_err.splitlines()[0]
    with pytest.raises(InstructionBuilderError) as e:
        parse_and_build(code)
    assert e.value.location is not None
    assert (
        get_location_marks(code, e.value.location) + "\n" + str(e.value.message)
        == code_with_err.rstrip()
    )
