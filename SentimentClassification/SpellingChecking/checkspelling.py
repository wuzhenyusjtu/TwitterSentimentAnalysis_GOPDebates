import enchant

fp = open('featurespace.txt','rb')
line = fp.readline()
feature_space = []
while line:
    word = line.strip()
    feature_space.append(word.upper())
    line = fp.readline()
fp.close()
feature_space = sorted(feature_space)

fpw = open('spelling.txt', 'wb')
d = enchant.Dict('en_US')
for word in feature_space:
    if not d.check(word):
        fpw.write(word.lower()+'\n')