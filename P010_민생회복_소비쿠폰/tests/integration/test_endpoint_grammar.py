"""Run in the SGLang environment; verifies its actual grammar accepts/rejects JSON."""
import copy
import json
from pathlib import Path
import sys
import unittest
import xgrammar as xgr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/sim'))
from planning_contract import endpoint_schedule_schema
from evidence_contract import constrain_evidence, constrain_field, constrain_trigger_evidence, ROUTINE


class EndpointGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'], stop_token_ids=[0]))
        cls.grammars = {weekend: compiler.compile_json_schema(endpoint_schedule_schema(['11680521'], weekend, True))
                        for weekend in [False, True]}

    def valid(self, count):
        return {'events': [{'time': f'{i+6:02d}:00', 'anchor': 'residence', 'category': '집',
                            'intent': '휴식', 'reasoning': '쉬기로 한다.', 'trigger': 'none'} for i in range(count)],
                'daily_propensity': .5}

    def accepts(self, value, weekend=False):
        matcher = xgr.GrammarMatcher(self.grammars[weekend])
        return matcher.accept_string(json.dumps(value, ensure_ascii=False)) and matcher.accept_token(0)

    def test_each_allowed_count_and_home_only_are_preserved(self):
        for weekend, counts in [(False, range(6, 11)), (True, range(4, 9))]:
            for count in counts:
                with self.subTest(weekend=weekend, count=count):
                    self.assertTrue(self.accepts(self.valid(count), weekend))

    def test_middle_outing_is_allowed_but_outside_endpoints_rejected(self):
        outside = {'time': '08:00', 'anchor': 'zone:11680521', 'category': '여가',
                   'intent': '산책', 'reasoning': '걷기로 한다.', 'trigger': 'lifestyle'}
        middle = self.valid(6); middle['events'][2] = outside
        self.assertTrue(self.accepts(middle))
        for index in [0, -1]:
            value = self.valid(6); value['events'][index] = copy.deepcopy(outside)
            self.assertFalse(self.accepts(value))

    def test_bad_count_and_anchor_rejected(self):
        self.assertFalse(self.accepts(self.valid(5)))
        self.assertFalse(self.accepts(self.valid(11)))
        value = self.valid(6); value['events'][2]['anchor'] = 'zone:99999999'
        self.assertFalse(self.accepts(value))

    def test_verbatim_evidence_grammar_rejects_new_factual_claim(self):
        schema = constrain_evidence(endpoint_schedule_schema(['11680521'], False, True), [ROUTINE, '독서를 즐긴다.'])
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'], stop_token_ids=[0]))
        grammar = compiler.compile_json_schema(schema)
        value = self.valid(6)
        for event in value['events']:
            event['reasoning'] = ROUTINE
        matcher = xgr.GrammarMatcher(grammar)
        self.assertTrue(matcher.accept_string(json.dumps(value, ensure_ascii=False)) and matcher.accept_token(0))
        value['events'][2]['reasoning'] = '매일 약을 복용한다.'
        self.assertFalse(xgr.GrammarMatcher(grammar).accept_string(json.dumps(value, ensure_ascii=False)))

    def test_bounded_whitespace_ebnf_preserves_values_and_blocks_loops(self):
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'], stop_token_ids=[0]))
        ebnf = str(xgr.Grammar.from_json_schema(endpoint_schedule_schema(['11680521'], False, True), max_whitespace_cnt=2))
        grammar = compiler.compile_grammar(ebnf)
        raw = json.dumps(self.valid(6), ensure_ascii=False)
        matcher = xgr.GrammarMatcher(grammar)
        self.assertTrue(matcher.accept_string(raw) and matcher.accept_token(0))
        self.assertFalse(xgr.GrammarMatcher(grammar).accept_string('{\n' + '  \n' * 30 + raw[1:]))

    def test_policy_label_cannot_cite_ordinary_employment_fact(self):
        schema = constrain_evidence(endpoint_schedule_schema(['11680521'], False, True), [ROUTINE, '직장이 있다.', '공공 이용 제한이 있다.'])
        schema = constrain_field(schema, 'trigger', ['none', 'policy'])
        schema = constrain_trigger_evidence(schema, {'policy': ['공공 이용 제한이 있다.']})
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'], stop_token_ids=[0]))
        grammar = compiler.compile_json_schema(schema)
        value = self.valid(6)
        for event in value['events']: event['reasoning'] = ROUTINE
        value['events'][2].update(trigger='policy', reasoning='공공 이용 제한이 있다.')
        matcher = xgr.GrammarMatcher(grammar)
        self.assertTrue(matcher.accept_string(json.dumps(value, ensure_ascii=False)) and matcher.accept_token(0))
        value['events'][2]['reasoning'] = '직장이 있다.'
        self.assertFalse(xgr.GrammarMatcher(grammar).accept_string(json.dumps(value, ensure_ascii=False)))
        value['events'][2]['trigger'] = 'none'
        self.assertTrue(xgr.GrammarMatcher(grammar).accept_string(json.dumps(value, ensure_ascii=False)))


if __name__ == '__main__':
    unittest.main()
