import json

from polehealth.register import REFERENCE_YEAR, YEAR_SHIFT, build_register, osm_poles, write_register


def test_same_seed_same_output(osm_payload, sheets):
    a = build_register(osm_payload, sheets, seed=7)
    b = build_register(osm_payload, sheets, seed=7)
    assert json.dumps(a) == json.dumps(b)


def test_different_seed_changes_pairing_or_attributes(osm_payload, sheets):
    a = build_register(osm_payload, sheets, seed=7)
    b = build_register(osm_payload, sheets, seed=8)
    assert json.dumps(a["poles"]) != json.dumps(b["poles"])


def test_osm_input_order_does_not_matter(osm_payload, sheets):
    shuffled = {"elements": list(reversed(osm_payload["elements"]))}
    assert build_register(osm_payload, sheets)["poles"] == build_register(shuffled, sheets)["poles"]


def test_poles_are_nodes_sorted_by_osm_id_without_duplicates(osm_payload):
    ids = [p["id"] for p in osm_poles(osm_payload)]
    assert ids == sorted(set(ids))
    assert len(ids) == 12  # the fixture has a way and a duplicate node mixed in


def test_year_shift_and_age(register):
    assert YEAR_SHIFT == 7
    for pole in register["poles"]:
        record = pole["record"]
        years = [i["year"] for i in record["inspections"]]
        assert years == [2006, 2016, 2026]
        assert record["install_year"] == 2026 - record["age_years"]
        assert "health_index" in record["inspections"][-1]
        assert all("health_index" not in i for i in record["inspections"][:-1])


def test_age_is_shifted_by_seven(osm_payload, sheets):
    register = build_register(osm_payload, sheets)
    for pole in register["poles"]:
        history_id = int(pole["record"]["source_history_id"].split("-")[1])
        assert pole["record"]["age_years"] == int(sheets[2019].loc[history_id]["Age"]) + 7


def test_record_fields_and_precomputed_numbers(register):
    for pole in register["poles"]:
        r = pole["record"]
        assert r["material"] == "timber"
        assert r["transformer"] in {"none", "single_phase", "three_phase"}
        assert r["bushfire_zone"] in {"high", "medium", "low"}
        assert 1 <= r["customers_served"] <= 400
        newest, previous = r["inspections"][-1]["shell_thickness"], r["inspections"][-2]["shell_thickness"]
        assert r["shell_thickness_change_10y"] == round(sum(newest) / 3 - sum(previous) / 3, 2)
        if r["last_maintained_year"] is None:
            assert r["years_since_maintenance"] is None
        else:
            assert 2016 <= r["last_maintained_year"] <= 2026
            assert r["years_since_maintenance"] == REFERENCE_YEAR - r["last_maintained_year"]
        assert (r["critical_customer_type"] is not None) == r["critical_customer"]
        assert isinstance(r["inspections"][0]["woodpecker_holes"], bool)


def test_histories_are_not_reused_when_there_are_enough(register):
    ids = [p["record"]["source_history_id"] for p in register["poles"]]
    assert len(ids) == len(set(ids))


def test_bushfire_split_follows_the_stated_shares(osm_payload, sheets):
    # 12 poles: 15% -> 2 high, 30% -> 4 medium, rest low
    zones = [p["record"]["bushfire_zone"] for p in build_register(osm_payload, sheets)["poles"]]
    assert (zones.count("high"), zones.count("medium"), zones.count("low")) == (2, 4, 6)


def test_write_register_round_trips(tmp_path, register):
    path = tmp_path / "out" / "register.json"
    write_register(register, path)
    assert json.loads(path.read_text()) == register
