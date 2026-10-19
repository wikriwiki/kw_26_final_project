# -*- coding: utf-8 -*-
"""사회 배경(환경) 채널.

정책과 독립한 그날의 세상 상태를 만든다. 정책이 없어도 존재하며,
수급자·비수급자 모두에게 같게 적용된다. 환경이 없으면 빈 dict 를 반환하고
프롬프트에서 섹션 자체가 생략된다.
"""
from .registry import build_environment, list_environments  # noqa: F401
