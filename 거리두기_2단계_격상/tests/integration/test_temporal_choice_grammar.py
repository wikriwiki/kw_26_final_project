import copy
import json
from pathlib import Path
import sys
import unittest
import xgrammar as xgr
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/sim'))
from temporal_choice_grammar import build


class TemporalGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell={'date':'2026-09-21','has_work':True,'zones':['Z'],'user':'09:00부터 17:00까지 사무실 근무.',
            'required_activities':[{'time':'09:00','activity_id':'office_work','anchor':'workplace','evidence':'09:00부터 17:00까지 사무실 근무.'}],
            'required_presence_intervals':[{'start':'09:00','end':'17:00','anchor':'workplace','evidence':'09:00부터 17:00까지 사무실 근무.'}],
            'minimum_transitions':[{'from_anchor':'residence','to_anchor':'workplace','minimum_minutes':40},{'from_anchor':'workplace','to_anchor':'residence','minimum_minutes':40}]}
        text,cls.audit=build(cls.cell,clock_step=30)
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'],stop_token_ids=[0]))
        cls.grammar=compiler.compile_grammar(text)

    def accepts(self,obj):
        m=xgr.GrammarMatcher(self.grammar)
        return m.accept_string(json.dumps(obj,ensure_ascii=False,separators=(',',':'))) and m.accept_token(0)

    def test_feasible_plan_and_invalid_start_or_order(self):
        obj=copy.deepcopy(self.audit['feasible_example']);self.assertTrue(self.accepts(obj))
        work=next(i for i,e in enumerate(obj['events']) if e['time']=='09:00')
        obj['events'][work]['time']='09:30';self.assertFalse(self.accepts(obj))
        obj=copy.deepcopy(self.audit['feasible_example']);obj['events'].reverse();self.assertFalse(self.accepts(obj))

    def test_committed_interval_cannot_have_midday_home_activity(self):
        events=[{'time':t,'activity_id':a,'anchor':p} for t,a,p in [
            ('07:00','home_prepare','residence'),('08:00','home_meal','residence'),('09:00','office_work','workplace'),
            ('14:00','home_chores','residence'),('17:00','office_finish','workplace'),('18:00','home_meal','residence')]]
        self.assertFalse(self.accepts({'events':events}))
        events[3].update(activity_id='office_break',anchor='workplace');self.assertTrue(self.accepts({'events':events}))
        events[-1]['time']='17:30';self.assertFalse(self.accepts({'events':events}))

    def test_explicit_coverage_boundary_rejects_early_ending(self):
        grammar,audit=build(self.cell,clock_step=60,last_start_not_before='20:00')
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo(['<eos>'],stop_token_ids=[0]))
        compiled=compiler.compile_grammar(grammar)
        def accepts(obj):
            matcher=xgr.GrammarMatcher(compiled)
            return matcher.accept_string(json.dumps(obj,separators=(',',':'))) and matcher.accept_token(0)
        obj=copy.deepcopy(audit['feasible_example']);self.assertTrue(accepts(obj))
        obj['events'][-1]['time']='19:00';self.assertFalse(accepts(obj))


if __name__=='__main__':unittest.main()
