"""The chat route for recorded-event questions: detection, figures straight from the engine, scope refusal outside the US, no LLM needed."""
from cri.resident import eventanswer as EA
from cri import eventlearn_tc as TC


def test_detects_wind_in_knots_mph_category():
    assert EA.detect("what total loss from a 115 kt hurricane in the US?")["wind_kt"] == 115
    assert EA.detect("how much damage from a 140 mph storm landfall")["wind_kt"] == 122
    d = EA.detect("what would a category 4 hurricane cost")
    assert d["wind_kt"] == 125 and "category 4" in d["note"]
    assert EA.detect("what is a hurricane") is None


def test_numbers_come_from_the_engine():
    r = EA.answer("what loss for a 115 knot hurricane in the US", use_llm=False)
    p = TC.predict(115)
    assert f"${p['expected_musd'] / 1000:.1f} billion" in r["reply"]
    assert "not an insured loss" in r["reply"] and "Katrina" in r["reply"]


def test_refuses_outside_scope_and_asks_for_wind():
    assert "not tested outside the US" in EA.answer("loss from a 110 kt typhoon in the Philippines", use_llm=False)["reply"]
    assert "peak wind" in EA.answer("tropical storm cost estimate", use_llm=False)["reply"]


def test_revenue_question_states_no_skill():
    r = EA.answer("will our revenue fall after a flood", use_llm=False)
    assert "not a forecast of your revenue" in r["reply"]
    r2 = EA.answer("will revenue fall after a drought", use_llm=False)
    assert "not assessed, not zero" in r2["reply"]
