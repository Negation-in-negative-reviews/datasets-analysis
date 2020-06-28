from pycorenlp import StanfordCoreNLP
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = (
        'i will not come back')
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse',
        'outputFormat': 'json'
    })
    myprint(output)
    # myprint(output['sentences'][0]["basicDependencies"])
    # myprint(output['sentences'][0]["enhancedDependencies"])
    # output = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
    # # print('dependency parse of first sentence')
    # # dependency_parse = sentence.basicDependencies
    # # print(dependency_parse)
    # output = nlp.semgrex(text, pattern='{tag: VBZ}', filter=False)
    # print(output)