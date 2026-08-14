// Neo4j Day 0 — UNIQUE 제약 + 인덱스
// 실행: cypher-shell -u neo4j -p <pwd> < 00_constraints.cypher

// =========================================================
// UNIQUE 제약
// =========================================================
CREATE CONSTRAINT agent_id        IF NOT EXISTS FOR (a:Agent)        REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT poi_id          IF NOT EXISTS FOR (p:POI)          REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT district_code   IF NOT EXISTS FOR (d:District)     REQUIRE d.code IS UNIQUE;
CREATE CONSTRAINT dong_code       IF NOT EXISTS FOR (d:Dong)         REQUIRE d.code IS UNIQUE;
CREATE CONSTRAINT category_name   IF NOT EXISTS FOR (c:Category)     REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT state_id        IF NOT EXISTS FOR (s:State)        REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT plan_id         IF NOT EXISTS FOR (p:Plan)         REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT memory_id       IF NOT EXISTS FOR (m:Memory)       REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT conv_id         IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT policy_id       IF NOT EXISTS FOR (p:Policy)       REQUIRE p.id IS UNIQUE;

// =========================================================
// 노드 속성 인덱스
// =========================================================
CREATE INDEX poi_type             IF NOT EXISTS FOR (p:POI)   ON (p.type);
CREATE INDEX poi_dong             IF NOT EXISTS FOR (p:POI)   ON (p.dong_code);
CREATE INDEX agent_gender         IF NOT EXISTS FOR (a:Agent) ON (a.p_gender);
CREATE INDEX agent_age_group      IF NOT EXISTS FOR (a:Agent) ON (a.p_age_group);
CREATE INDEX agent_income         IF NOT EXISTS FOR (a:Agent) ON (a.p_income_level);
CREATE INDEX agent_life_stage     IF NOT EXISTS FOR (a:Agent) ON (a.p_life_stage);
CREATE INDEX agent_lifestyle      IF NOT EXISTS FOR (a:Agent) ON (a.pr_lifestyle_cluster);

CREATE INDEX state_agent_day      IF NOT EXISTS FOR (s:State)        ON (s.agent_id, s.day);
CREATE INDEX plan_agent_day       IF NOT EXISTS FOR (p:Plan)         ON (p.agent_id, p.day);
CREATE INDEX memory_day_type      IF NOT EXISTS FOR (m:Memory)       ON (m.day, m.type);
CREATE INDEX conv_intent_day      IF NOT EXISTS FOR (c:Conversation) ON (c.intent, c.day);
// Dawn ④ 약속 자동 조회 — should_inject + target_day_offset 결합으로 today 매칭
CREATE INDEX conv_inject_offset   IF NOT EXISTS FOR (c:Conversation) ON (c.should_inject, c.target_day_offset);
// recipient별 일자 조회 (Night Phase 3, Memory join 등)
CREATE INDEX conv_recipient_day   IF NOT EXISTS FOR (c:Conversation) ON (c.recipient_id, c.day);
CREATE INDEX policy_effective     IF NOT EXISTS FOR (p:Policy)       ON (p.effective_from, p.effective_until);

// =========================================================
// 관계 속성 인덱스 (Dawn 7종 Cypher 성능)
// =========================================================
CREATE INDEX rel_has_state_day    IF NOT EXISTS FOR ()-[r:HAS_STATE]-() ON (r.day);
CREATE INDEX rel_has_plan_day     IF NOT EXISTS FOR ()-[r:HAS_PLAN]-()  ON (r.day);
CREATE INDEX rel_remembers_day    IF NOT EXISTS FOR ()-[r:REMEMBERS]-() ON (r.day);
CREATE INDEX rel_includes_order   IF NOT EXISTS FOR ()-[r:INCLUDES]-()  ON (r.order);
