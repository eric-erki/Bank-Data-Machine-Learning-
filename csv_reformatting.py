
rt = 'bank-crossvalidation.csv'
nrt = 'bank-crossvalidation_new.csv'
fi = open(rt, 'r')
fo = open(nrt, 'w+')

#"58;""management"";""married"";""tertiary"";""no"";2143;""yes"";""no"";""unknown"";5;""may"";261;1;-1;0;""unknown"";""no"""


for line in fi.readlines():
    print(line)
    line = line.replace('\"\";\"\"', ',')
    line = line.replace(';\"\"', ',')
    line = line.replace('\"\";', ',')
    line = line.replace(';', ',')
    line = line.replace('\"\"\"', '')
    line = line.replace('\"', '')
    print(line)
    fo.writelines(line)

fi.close()
fo.close()

