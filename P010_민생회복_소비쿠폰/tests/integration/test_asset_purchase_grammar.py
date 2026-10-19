"""Use the actual installed XGrammar compiler, without model calls."""
import copy
import json
from pathlib import Path
import sys
import unittest
import xgrammar as xgr
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/sim'))
from asset_transaction_contract_v2 import schema


class PurchaseGrammarTest(unittest.TestCase):
    def setUp(self):
        self.case={'cash':10000,'wallet_lots':{},'offers':{},'events':[
            {'id':'A','channel':'offline','candidates':[]},
            {'id':'B','channel':'offline','candidates':[{'id':'Q','price_won':4000,'eligible_wallets':[]}]}]}
        self.obj={'acquisitions':[],'purchases':[{'id':i,'candidate_id':None,'cash_payment':0,'wallet_spend':{},'reason':'선택 없음'} for i in ['A','B']]}

    def accepts(self,case,obj):
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'],stop_token_ids=[0]))
        grammar=compiler.compile_grammar(str(xgr.Grammar.from_json_schema(schema(case),max_whitespace_cnt=2)))
        matcher=xgr.GrammarMatcher(grammar)
        return matcher.accept_string(json.dumps(obj,ensure_ascii=False)) and matcher.accept_token(0)

    def test_zero_and_positive_choices_allowed_but_order_not_optional(self):
        self.assertTrue(self.accepts(self.case,self.obj))
        self.obj['purchases'][1].update(candidate_id='Q',cash_payment=4000)
        self.assertTrue(self.accepts(self.case,self.obj))
        self.obj['purchases'].reverse();self.assertFalse(self.accepts(self.case,self.obj))

    def test_no_duplicate_or_missing_event(self):
        self.obj['purchases'][1]=copy.deepcopy(self.obj['purchases'][0])
        self.assertFalse(self.accepts(self.case,self.obj))
        self.obj['purchases'].pop();self.assertFalse(self.accepts(self.case,self.obj))

    def test_empty_purchase_day_can_acquire_asset(self):
        self.case['events']=[];self.case['offers']={'O':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':1}}
        obj={'acquisitions':[{'offer_id':'O','units':1,'reason':'나중에 사용','before_event_id':None}],'purchases':[]}
        self.assertTrue(self.accepts(self.case,obj))

    def test_v3_funding_and_no_purchase_choices_without_redundant_cash(self):
        from asset_transaction_contract_v3 import schema as schema3
        c={'cash':10000,'wallet_lots':{'V':[{'face':5000,'own_basis':0}]},'offers':{},
           'events':[{'id':'A','channel':'offline','candidates':[{'id':'Q','price_won':2500,'eligible_wallets':['V']}]}]}
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'],stop_token_ids=[0]))
        grammar=compiler.compile_grammar(str(xgr.Grammar.from_json_schema(schema3(c),max_whitespace_cnt=2)))
        def accepts(obj):
            m=xgr.GrammarMatcher(grammar);return m.accept_string(json.dumps(obj)) and m.accept_token(0)
        obj={'acquisition_units':{},'purchases':[{'id':'A','candidate_id':'Q','wallet_spend':{'V':2500}}]}
        self.assertTrue(accepts(obj))
        obj['purchases'][0]['cash_payment']=2500;self.assertFalse(accepts(obj))
        obj['purchases'][0].pop('cash_payment');obj['purchases'][0]['wallet_spend']={'UNKNOWN':2500};self.assertFalse(accepts(obj))
        obj['purchases'][0].update(candidate_id=None,wallet_spend={});self.assertTrue(accepts(obj))

    def test_v4_acquisition_zero_does_not_make_wallet_available(self):
        from asset_transaction_contract_v4 import schema as schema4
        c={'cash':9000,'wallet_lots':{},'offers':{'O':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':50}},
           'execution_assumptions':{'offers_available_before_any_purchase':True,'intraday_income':0},
           'events':[{'id':'A','channel':'offline','candidates':[{'id':'Q','price_won':2500,'eligible_wallets':['V']}]}]}
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'],stop_token_ids=[0]))
        grammar=compiler.compile_grammar(str(xgr.Grammar.from_json_schema(schema4(c),max_whitespace_cnt=2)))
        def accepts(obj):
            m=xgr.GrammarMatcher(grammar);return m.accept_string(json.dumps(obj)) and m.accept_token(0)
        obj={'acquisition_units':{'O':0},'purchases':[{'id':'A','candidate_id':'Q','wallet_spend':{'V':2500}}]}
        self.assertFalse(accepts(obj))
        obj['acquisition_units']['O']=1;self.assertTrue(accepts(obj))
        obj['acquisition_units']['O']=2;self.assertFalse(accepts(obj))
        obj['acquisition_units']['O']=0;obj['purchases'][0]['wallet_spend']={};self.assertTrue(accepts(obj))


if __name__=='__main__':unittest.main()
