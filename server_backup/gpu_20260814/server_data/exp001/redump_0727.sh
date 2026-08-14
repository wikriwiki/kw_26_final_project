#!/bin/bash
NEO=/data/neo4j-community-5.26.0
ARCH=/data/exp001/archive
LOG=/data/exp001/redump_0727.log
echo "[$(date '+%m-%d %H:%M:%S')] 재덤프 시작" >> $LOG
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/POL7500H_p010_7d_FIXED.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
md5sum "$ARCH/POL7500H_p010_7d_FIXED.dump" >> $LOG 2>&1
ls -l "$ARCH/POL7500H_p010_7d_FIXED.dump" >> $LOG 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] 재덤프 완료" >> $LOG
