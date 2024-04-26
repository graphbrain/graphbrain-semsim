from graphbrain.parsers import create_parser
from graphbrain.patterns import match_pattern

parser = create_parser(lang='en')

text1 = ("Ann plays the lead melody in an opera")


parses = parser.parse(text1)['parses']
for parse in parses:
    edge1 = parse['main_edge']
    print(edge1.to_str())


text2 = ("Ann plays piano")


parses = parser.parse(text2)['parses']
for parse in parses:
    edge2 = parse['main_edge']
    print(edge2.to_str())

pattern1 = 'plays/P ann/C */C *'

pattern2 = 'plays/P ann/C */C'

result1 =  match_pattern(edge1, pattern1)
print(result1)

result2 =  match_pattern(edge2, pattern1)
print(result2)


result3 =  match_pattern(edge1, pattern2)
print(result3)

result4 =  match_pattern(edge2, pattern2)
print(result4)