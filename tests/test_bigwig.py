from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyBigWig
import pytest

from gpn_star_scores.bigwig import (
    BASES,
    BigWigValidationError,
    ChromosomeSpec,
    calibrated_llr_logo_heights,
    validate_bigwig,
    write_entropy_bigwig,
    write_entropy_wig,
    write_logo_bigwigs,
    write_logo_wigs,
)


def test_calibrated_llr_logo_heights_are_stable_float32() -> None:
    heights = calibrated_llr_logo_heights("C", {"A": -1.0e30, "G": 1.0e30, "T": 0.5})

    assert tuple(heights) == BASES
    assert all(value.dtype == np.float32 for value in heights.values())
    assert all(np.isfinite(value) for value in heights.values())
    assert heights["G"] == pytest.approx(2.0)
    assert heights["A"] == pytest.approx(0.0)


def test_equal_llrs_produce_zero_information() -> None:
    heights = calibrated_llr_logo_heights("A", {"C": 0.0, "G": 0.0, "T": 0.0})

    assert list(heights.values()) == [pytest.approx(0.0)] * 4


def test_logo_requires_exact_non_reference_alternates() -> None:
    with pytest.raises(BigWigValidationError, match="non-reference"):
        calibrated_llr_logo_heights("A", {"A": 1.0, "C": 2.0, "G": 3.0})


def test_entropy_writer_converts_coordinates_and_preserves_gaps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "entropy.parquet"
    output = tmp_path / "entropy.bw"
    table = pa.table(
        {
            "chrom": pa.array(["1", "1", "1"]),
            "pos": pa.array([1, 2, 5], type=pa.int64()),
            "ref": pa.array(["A", "C", "G"]),
            "entropy_calibrated": pa.array([0.25, 1.5, 0.75], type=pa.float32()),
        }
    )
    pq.write_table(table, source)

    stats = write_entropy_bigwig(
        [source],
        output,
        ChromosomeSpec("1", "chr1", 5),
        batch_size=2,
        header_chromosome_sizes={"chr1": 5, "chr2": 7},
    )

    assert stats.position_count == 3
    assert (stats.first_position, stats.last_position) == (1, 5)
    summary = validate_bigwig(
        output,
        {"chr1": 5, "chr2": 7},
        expected_bases_covered=3,
    )
    assert summary.bases_covered == 3
    assert summary.zoom_levels >= 1
    with pyBigWig.open(str(output)) as bigwig:
        values = bigwig.values("chr1", 0, 5)
    assert values[:2] == pytest.approx([0.25, 1.5])
    assert np.isnan(values[2:4]).all()
    assert values[4] == pytest.approx(0.75)


def test_logo_writer_handles_alt_order_gaps_and_split_positions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "llr.parquet"
    outputs = {base: tmp_path / f"{base}.bw" for base in BASES}
    table = pa.table(
        {
            "chrom": pa.array(["1"] * 9),
            "pos": pa.array([1, 1, 1, 2, 2, 2, 5, 5, 5], type=pa.int64()),
            "ref": pa.array(["A"] * 3 + ["C"] * 3 + ["T"] * 3),
            "alt": pa.array(["T", "C", "G", "G", "A", "T", "C", "G", "A"]),
            "llr_calibrated": pa.array(
                [0.1, 0.2, 0.3, -0.5, 1.0, 0.0, 0.4, -0.2, 0.8],
                type=pa.float32(),
            ),
            "abs_llr_calibrated": pa.array([99.0] * 9, type=pa.float32()),
        }
    )
    pq.write_table(table, source, row_group_size=4)

    stats = write_logo_bigwigs(
        [source], outputs, ChromosomeSpec("1", "chr1", 5), batch_size=4
    )

    assert stats.position_count == 3
    expected = calibrated_llr_logo_heights("C", {"G": -0.5, "A": 1.0, "T": 0.0})
    for base, output in outputs.items():
        with pyBigWig.open(str(output)) as bigwig:
            values = bigwig.values("chr1", 0, 5)
        assert values[1] == pytest.approx(float(expected[base]), abs=1e-7)
        assert np.isnan(values[2:4]).all()


def test_variable_step_wig_baselines_preserve_float32_values(tmp_path: Path) -> None:
    entropy_source = tmp_path / "entropy.parquet"
    entropy_wig = tmp_path / "entropy.wig"
    entropy_values = np.array([np.float32(1.234567), np.float32(0.00012345678)])
    pq.write_table(
        pa.table(
            {
                "chrom": pa.array(["1", "1"]),
                "pos": pa.array([1, 5], type=pa.int64()),
                "ref": pa.array(["A", "T"]),
                "entropy_calibrated": pa.array(entropy_values),
            }
        ),
        entropy_source,
    )

    write_entropy_wig([entropy_source], entropy_wig, ChromosomeSpec("1", "chr1", 5))

    entropy_lines = entropy_wig.read_text().splitlines()
    assert entropy_lines[0] == "variableStep chrom=chr1 span=1"
    assert [int(line.split("\t")[0]) for line in entropy_lines[1:]] == [1, 5]
    round_tripped = np.array(
        [np.float32(line.split("\t")[1]) for line in entropy_lines[1:]]
    )
    np.testing.assert_array_equal(round_tripped, entropy_values)

    llr_source = tmp_path / "llr.parquet"
    logo_outputs = {base: tmp_path / f"{base}.wig" for base in BASES}
    pq.write_table(
        pa.table(
            {
                "chrom": pa.array(["1"] * 3),
                "pos": pa.array([5, 5, 5], type=pa.int64()),
                "ref": pa.array(["T"] * 3),
                "alt": pa.array(["A", "C", "G"]),
                "llr_calibrated": pa.array([0.1, 0.2, 0.3], type=pa.float32()),
            }
        ),
        llr_source,
    )

    write_logo_wigs(
        [llr_source], logo_outputs, ChromosomeSpec("1", "chr1", 5), batch_size=2
    )

    expected = calibrated_llr_logo_heights(
        "T", {"A": np.float32(0.1), "C": np.float32(0.2), "G": np.float32(0.3)}
    )
    for base, output in logo_outputs.items():
        position, value = output.read_text().splitlines()[1].split("\t")
        assert int(position) == 5
        assert np.float32(value) == expected[base]


def test_entropy_writer_rejects_out_of_order_positions_without_replacing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "entropy.parquet"
    output = tmp_path / "entropy.bw"
    output.write_bytes(b"existing artifact")
    table = pa.table(
        {
            "chrom": pa.array(["1", "1"]),
            "pos": pa.array([2, 1], type=pa.int64()),
            "ref": pa.array(["C", "A"]),
            "entropy_calibrated": pa.array([0.25, 0.5], type=pa.float32()),
        }
    )
    pq.write_table(table, source)

    with pytest.raises(BigWigValidationError, match="strictly increasing"):
        write_entropy_bigwig([source], output, ChromosomeSpec("1", "chr1", 2))

    assert output.read_bytes() == b"existing artifact"
    assert not list(tmp_path.glob(".entropy.bw.*.tmp"))


def test_logo_writer_rejects_incomplete_position(tmp_path: Path) -> None:
    source = tmp_path / "llr.parquet"
    table = pa.table(
        {
            "chrom": pa.array(["1", "1"]),
            "pos": pa.array([1, 1], type=pa.int64()),
            "ref": pa.array(["A", "A"]),
            "alt": pa.array(["C", "G"]),
            "llr_calibrated": pa.array([0.1, 0.2], type=pa.float32()),
        }
    )
    pq.write_table(table, source)

    with pytest.raises(BigWigValidationError, match="exactly three"):
        write_logo_bigwigs(
            [source],
            {base: tmp_path / f"{base}.bw" for base in BASES},
            ChromosomeSpec("1", "chr1", 2),
        )
