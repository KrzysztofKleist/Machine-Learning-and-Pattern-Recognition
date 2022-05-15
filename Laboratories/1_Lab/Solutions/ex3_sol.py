import sys

hMonthNames = {
    1: 'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November',
    12:'December'}

if __name__ == '__main__':

    try:
        f = open(sys.argv[1], 'r')
    except:
        print("Error opening the file")
        sys.exit(1)

    hCities = {}
    hMonths = {}

    for line in f:

        name, surname, city, date = line.split()
        day, month, year = date.split('/')
        month_int = int(month)

        if city not in hCities:
            hCities[city] = 0
        hCities[city] += 1
        
        if month_int not in hMonths:
            hMonths[month_int] = 0
        hMonths[month_int] += 1

    f.close()

    print('Births per city:')
    for city in hCities:
        print('\t%s: %d' % (city, hCities[city]))
    print('Births per month:')
    for month in sorted(hMonths):
        print('\t%s: %d' % (hMonthNames[month], hMonths[month]))
    print('Average number of births: %.2f' % (float(sum(hCities.values()))/float(len(hCities.keys()))))
              
            
