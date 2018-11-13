import contractions
import nltk

test = """
Email
Person holding american flag. (iStock)
We celebrate the Fourth of July because that's the day the Declaration of Independence was signed, 242 years ago. You might call July 4 America's birthday.
The Declaration didn't just declare our independence from Britain; it vowed to create a government that respected all people's rights to "life, liberty and the pursuit of happiness."
It said nothing about patriotism. Or making America "great."
America became great because the Declaration (and the Constitution that followed) set down rules that kept government small and out of the way. That let creative individuals flourish.
When the Declaration was signed, the founders didn't know what America would look like. They knew, though, that they were sick of being bossed around by the British king, so they worried about government having too much power.
Thomas Jefferson and his colleagues wrote the Declaration to assert that our "natural rights" could not be taken away by any government, and to set the stage for the creation of a government through which people could rule themselves.
At the time, America was considered a backwater. Just a few years later, America had become the most prosperous, and probably the freest, country in the world.
The Fourth of July is not about barbecues, fireworks or even patriotism. It's about that idea: that people have the right to rule themselves.
Ironically, government has grown so much since the founding that you might not even be able to buy fireworks where you live. In much of America they are now illegal because government officials have declared them to be too dangerous.
Yet the Declaration and Constitution weren't written to make government provide for public safety. The founders assumed that was something adults would do for themselves. The founding documents are about freedom -- about limiting what government can do.
"Trust no man/with too much government power," wrote Jefferson. "(B)ind them with the chains of the Constitution."
It's good that the Declaration and Constitution have those "chains." No matter how insistent the state's busybodies get, they may not arbitrarily search our homes or jail us. We have a right to bear arms, to practice whatever religions we choose, to exercise free speech and more.
Growing government has eroded some of our freedoms, but we still have more freedoms than most countries in the world.
Consider the country we declared independence from, Great Britain. Authorities there recently locked up a man merely because he made a Facebook live video outside a courthouse. He wanted to draw attention to child abusers on trial, but Britain's government puts limits on what reporters may cover. England has no First Amendment.
Some people who write critical things on Facebook or Twitter get visits from police.
Great Britain also has no Second Amendment, and has far more restrictions on guns than we have. That hasn't stopped crime. London had more murders than New York City this spring.
Now London's mayor wants "knife control." Really.
One British police department even www.google.de bragged about its "weapon sweep" that confiscated "scissors and pliers." But don't worry, tweeted the Regents Police Agency, they were "safely disposed and taken off the streets."
I'm glad I live in America, where I can carry pliers around. And speak freely .
Of course, the Constitution has more limits on government power than just the ones stated in the Bill of Rights.
The Constitution divided government in ways meant to limit authoritarian politicians from any party.
President Donald Trump's own Supreme Court nominee rebuked the man who appointed him, ruling that a Trump-advocated law making it easier to deport some immigrants was too vague.
The Court stopped President Obama almost a hundred times.
It's a good thing we have both the Declaration and the Constitution, with their curbs on power-grabs by presidents and legislators -- curbs on judges, too.
Unfortunately, those limits on government haven't exactly kept government small. Thomas Jefferson wanted "a wise and frugal government" that leaves people "free to regulate their own pursuits." Now we've got 180,000 pages of federal rules and $21 trillion in debt.
Still, the Constitution and the Declaration have helped keep us mostly free. That's something to celebrate this Fourth of July.
"""

print([token for token in tokens if token.isalpha()])

['Email', 'Person', 'holding', 'american', 'flag', 'iStock', 'We', 'celebrate', 'the', 'Fourth', 'of', 'July',
 'because', 'that', 'is', 'the', 'day', 'the', 'Declaration', 'of', 'Independence', 'was', 'signed', 'years', 'ago',
 'You', 'might', 'call', 'July', 'America', 'birthday', 'The', 'Declaration', 'did', 'not', 'just', 'declare', 'our',
 'independence', 'from', 'Britain', 'it', 'vowed', 'to', 'create', 'a', 'government', 'that', 'respected', 'all',
 'people', 'rights', 'to', 'life', 'liberty', 'and', 'the', 'pursuit', 'of', 'happiness', 'It', 'said', 'nothing',
 'about', 'patriotism', 'Or', 'making', 'America', 'great', 'America', 'became', 'great', 'because', 'the',
 'Declaration', 'and', 'the', 'Constitution', 'that', 'followed', 'set', 'down', 'rules', 'that', 'kept', 'government',
 'small', 'and', 'out', 'of', 'the', 'way', 'That', 'let', 'creative', 'individuals', 'flourish', 'When', 'the',
 'Declaration', 'was', 'signed', 'the', 'founders', 'did', 'not', 'know', 'what', 'America', 'would', 'look', 'like',
 'They', 'knew', 'though', 'that', 'they', 'were', 'sick', 'of', 'being', 'bossed', 'around', 'by', 'the', 'British',
 'king', 'so', 'they', 'worried', 'about', 'government', 'having', 'too', 'much', 'power', 'Thomas', 'Jefferson', 'and',
 'his', 'colleagues', 'wrote', 'the', 'Declaration', 'to', 'assert', 'that', 'our', 'natural', 'rights', 'could', 'not',
 'be', 'taken', 'away', 'by', 'any', 'government', 'and', 'to', 'set', 'the', 'stage', 'for', 'the', 'creation', 'of',
 'a', 'government', 'through', 'which', 'people', 'could', 'rule', 'themselves', 'At', 'the', 'time', 'America', 'was',
 'considered', 'a', 'backwater', 'Just', 'a', 'few', 'years', 'later', 'America', 'had', 'become', 'the', 'most',
 'prosperous', 'and', 'probably', 'the', 'freest', 'country', 'in', 'the', 'world', 'The', 'Fourth', 'of', 'July', 'is',
 'not', 'about', 'barbecues', 'fireworks', 'or', 'even', 'patriotism', 'it', 'is', 'about', 'that', 'idea', 'that',
 'people', 'have', 'the', 'right', 'to', 'rule', 'themselves', 'Ironically', 'government', 'has', 'grown', 'so', 'much',
 'since', 'the', 'founding', 'that', 'you', 'might', 'not', 'even', 'be', 'able', 'to', 'buy', 'fireworks', 'where',
 'you', 'live', 'In', 'much', 'of', 'America', 'they', 'are', 'now', 'illegal', 'because', 'government', 'officials',
 'have', 'declared', 'them', 'to', 'be', 'too', 'dangerous', 'Yet', 'the', 'Declaration', 'and', 'Constitution', 'were',
 'not', 'written', 'to', 'make', 'government', 'provide', 'for', 'public', 'safety', 'The', 'founders', 'assumed',
 'that', 'was', 'something', 'adults', 'would', 'do', 'for', 'themselves', 'The', 'founding', 'documents', 'are',
 'about', 'freedom', 'about', 'limiting', 'what', 'government', 'can', 'do', 'Trust', 'no', 'man', 'with', 'too',
 'much', 'government', 'power', 'wrote', 'Jefferson', 'B', 'ind', 'them', 'with', 'the', 'chains', 'of', 'the',
 'Constitution', 'it', 'is', 'good', 'that', 'the', 'Declaration', 'and', 'Constitution', 'have', 'those', 'chains',
 'No', 'matter', 'how', 'insistent', 'the', 'state', 'busybodies', 'get', 'they', 'may', 'not', 'arbitrarily', 'search',
 'our', 'homes', 'or', 'jail', 'us', 'We', 'have', 'a', 'right', 'to', 'bear', 'arms', 'to', 'practice', 'whatever',
 'religions', 'we', 'choose', 'to', 'exercise', 'free', 'speech', 'and', 'more', 'Growing', 'government', 'has',
 'eroded', 'some', 'of', 'our', 'freedoms', 'but', 'we', 'still', 'have', 'more', 'freedoms', 'than', 'most',
 'countries', 'in', 'the', 'world', 'Consider', 'the', 'country', 'we', 'declared', 'independence', 'from', 'Great',
 'Britain', 'Authorities', 'there', 'recently', 'locked', 'up', 'a', 'man', 'merely', 'because', 'he', 'made', 'a',
 'Facebook', 'live', 'video', 'outside', 'a', 'courthouse', 'He', 'wanted', 'to', 'draw', 'attention', 'to', 'child',
 'abusers', 'on', 'trial', 'but', 'Britain', 'government', 'puts', 'limits', 'on', 'what', 'reporters', 'may', 'cover',
 'England', 'has', 'no', 'First', 'Amendment', 'Some', 'people', 'who', 'write', 'critical', 'things', 'on', 'Facebook',
 'or', 'Twitter', 'get', 'visits', 'from', 'police', 'Great', 'Britain', 'also', 'has', 'no', 'Second', 'Amendment',
 'and', 'has', 'far', 'more', 'restrictions', 'on', 'guns', 'than', 'we', 'have', 'That', 'has', 'not', 'stopped',
 'crime', 'London', 'had', 'more', 'murders', 'than', 'New', 'York', 'City', 'this', 'spring', 'Now', 'London', 'mayor',
 'wants', 'knife', 'control', 'Really', 'One', 'British', 'police', 'department', 'even', 'bragged', 'about', 'its',
 'weapon', 'sweep', 'that', 'confiscated', 'scissors', 'and', 'pliers', 'But', 'do', 'not', 'worry', 'tweeted', 'the',
 'Regents', 'Police', 'Agency', 'they', 'were', 'safely', 'disposed', 'and', 'taken', 'off', 'the', 'streets', 'I',
 'am', 'glad', 'I', 'live', 'in', 'America', 'where', 'I', 'can', 'carry', 'pliers', 'around', 'And', 'speak', 'freely',
 'Of', 'course', 'the', 'Constitution', 'has', 'more', 'limits', 'on', 'government', 'power', 'than', 'just', 'the',
 'ones', 'stated', 'in', 'the', 'Bill', 'of', 'Rights', 'The', 'Constitution', 'divided', 'government', 'in', 'ways',
 'meant', 'to', 'limit', 'authoritarian', 'politicians', 'from', 'any', 'party', 'President', 'Donald', 'Trump', 'own',
 'Supreme', 'Court', 'nominee', 'rebuked', 'the', 'man', 'who', 'appointed', 'him', 'ruling', 'that', 'a', 'Trump',
 'advocated', 'law', 'making', 'it', 'easier', 'to', 'deport', 'some', 'immigrants', 'was', 'too', 'vague', 'The',
 'Court', 'stopped', 'President', 'Obama', 'almost', 'a', 'hundred', 'times', 'it', 'is', 'a', 'good', 'thing', 'we',
 'have', 'both', 'the', 'Declaration', 'and', 'the', 'Constitution', 'with', 'their', 'curbs', 'on', 'power', 'grabs',
 'by', 'presidents', 'and', 'legislators', 'curbs', 'on', 'judges', 'too', 'Unfortunately', 'those', 'limits', 'on',
 'government', 'have', 'not', 'exactly', 'kept', 'government', 'small', 'Thomas', 'Jefferson', 'wanted', 'a', 'wise',
 'and', 'frugal', 'government', 'that', 'leaves', 'people', 'free', 'to', 'regulate', 'their', 'own', 'pursuits', 'Now',
 'we', 'have', 'got', 'pages', 'of', 'federal', 'rules', 'and', 'trillion', 'in', 'debt', 'Still', 'the',
 'Constitution', 'and', 'the', 'Declaration', 'have', 'helped', 'keep', 'us', 'mostly', 'free', 'that', 'is',
 'something', 'to', 'celebrate', 'this', 'Fourth', 'of', 'July']
