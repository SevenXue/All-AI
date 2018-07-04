import fasttext

classifier = fasttext.classifier
result = classifier.test('sms_0000_label_test.txt')
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples