import sys

def compute_avg_score(lscores):
    return sum(sorted(lscores)[1:-1])

class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.avg_score = compute_avg_score(self.scores)


if __name__ == '__main__':

    l_bestCompetitors = []
    hCountryScores = {}
    with open(sys.argv[1]) as f:
        for line in f:
            name, surname, country = line.split()[0:3]
            scores = line.split()[3:]
            scores = [float(i) for i in scores]
            comp = Competitor(
                name,
                surname,
                country,
                scores)
            l_bestCompetitors.append(comp)
            if len(l_bestCompetitors) >= 4:
                l_bestCompetitors = sorted(l_bestCompetitors, key = lambda i: i.avg_score)[::-1][0:3]
            if comp.country not in hCountryScores:
                hCountryScores[comp.country] = 0
            hCountryScores[comp.country] += comp.avg_score

    if len(hCountryScores) == 0:
        print('No competitors')
        sys.exit(0)

    best_country = None
    for count in hCountryScores:
        if best_country is None or hCountryScores[count] > hCountryScores[best_country]:
            best_country = count

    print('Final ranking:')
    for pos, comp in enumerate(l_bestCompetitors):
        print('%d: %s %s - Score: %.1f' % (pos+1, comp.name, comp.surname, comp.avg_score))
    print()
    print('Best Country:')
    print("%s - Total score: %.1f" % (best_country, hCountryScores[best_country]))
