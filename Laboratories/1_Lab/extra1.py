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


class BookRecord:
    def __init__(self, bookID, buySell, date, numOfCopies, price):
        self.bookID = bookID
        self.buySell = buySell
        self.date = hMonthNames[int(date.split('/')[1])] + ', ' + date.split('/')[2]
        self.numOfCopies = numOfCopies
        self.price = price


def loadAllRecords(fName):
    try:
        lRecords = []
        with open(fName) as f:
            for line in f:
                bookID, buySell, date, numOfCopies, price = line.split()
                record = BookRecord(bookID, buySell, date, numOfCopies, price)
                lRecords.append(record)
            return lRecords
    except:
        raise


def availableCopies(lRecords):
    hAvailable = {}
    for record in lRecords:
        if record.bookID not in hAvailable:
            hAvailable[record.bookID] = 0
        if record.buySell == 'B':
            hAvailable[record.bookID] += int(record.numOfCopies)
        else:
            hAvailable[record.bookID] -= int(record.numOfCopies)
    print('Available copies:')
    for book in hAvailable:
        print('\t%s: %d' % (book, hAvailable[book]))


def soldBooksMonth(lRecords):
    hMonths = {}
    for record in lRecords:
        if record.buySell == 'S':
            if record.date not in hMonths:
                hMonths[record.date] = 0
            hMonths[record.date] += int(record.numOfCopies)
    print('Books sold per month:')
    for month in hMonths:
        print('\t%s: %d' % (month, hMonths[month]))


def gainPerBook(lRecords):
    hGain = {}
    hNumSold = {}
    for record in lRecords:
        if record.bookID not in hGain:
            hGain[record.bookID] = 0.0
        if record.buySell == 'B':
            hGain[record.bookID] += float(float(record.numOfCopies) * float(record.price))
        else:
            hGain[record.bookID] -= float(float(record.numOfCopies) * float(record.price))
        if record.buySell == 'S':
            if record.bookID not in hNumSold:
                hNumSold[record.bookID] = 0
            hNumSold[record.bookID] += int(record.numOfCopies)
    print('Gain per book:')
    for book in hGain:
        print('\t%s: %.2f (avg: %.2f, sold: %d)' % (book, hGain[book], hGain[book]/hNumSold[book], hNumSold[book]))


if __name__ == '__main__':
    lRecords = loadAllRecords(sys.argv[1])
    availableCopies(lRecords)
    soldBooksMonth(lRecords)
    gainPerBook(lRecords)
