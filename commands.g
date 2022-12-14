<Action> + <Location>
<Condition> + <Location>

Action := "Jump"
Location := <Number> | <Number> + <Advanced location> | <Advanced location> | "gingival margin"
Number:= 1..32
Advanced location := "mesial" | "distal" | "all"
Condition:= "bleeding" | "furcation <Location> grade <Number>" | "plaque <light|medium|severe>"