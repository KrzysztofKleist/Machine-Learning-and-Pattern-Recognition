import sys

hMonthNames = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}


class BirthRecord:
    def __init__(self, name, surname, city, date):
        self.name = name
        self.surname = surname
        self.city = city
        self.month = date.split('/')[1]


def loadAllRecords(fName):
    try:
        lRecords = []
        with open(fName) as f:
            for line in f:
                name, surname, city, date = line.split()
                record = BirthRecord(name, surname, city, date)
                lRecords.append(record)
            return lRecords
    except:
        raise


def computeBirthCity(lRecords):
    hCities = {}
    for record in lRecords:
        if record.city not in hCities:
            hCities[record.city] = 0
        hCities[record.city] += 1
    print('Births per city: ')
    for city in hCities:
        print('\t%s: %d' % (city, hCities[city]))
    print('Average number of births: %.2f' % (float(sum(hCities.values()))/float(len(hCities))))
    print()

def computeBirthMonth(lRecords):
    hMonths = {}
    for record in lRecords:
        if record.month not in hMonths:
            hMonths[record.month] = 0
        hMonths[record.month] += 1
    print('Births per month: ')
    for month in hMonths:
        print('\t%s: %d' % (hMonthNames[int(month)], hMonths[month]))


if __name__ == '__main__':

    lRecords = loadAllRecords(sys.argv[1])
    computeBirthCity(lRecords)
    computeBirthMonth(lRecords)


