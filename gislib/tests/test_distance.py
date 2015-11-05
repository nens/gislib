from unittest import TestCase

from gislib import vectors


class TestGreatCircleDistance(TestCase):
    def setUp(self):
        self.expected_dist = 577.35833
        self.coords = [(4.8896900, 52.3740300),
                       (13.4105300, 52.5243700)]

    def test_haversine_formula(self):
        dist_haversine = vectors.calculate_great_circle_distance(self.coords)
        self.assertAlmostEqual(dist_haversine, self.expected_dist, places=5)

    def test_cosine_formula(self):
        dist_cosine = vectors.calculate_great_circle_distance(
            self.coords, formula='cosine'
        )
        self.assertAlmostEqual(dist_cosine, self.expected_dist, places=5)

    def test_cosine_vs_haversine(self):
        dist_haversine = vectors.calculate_great_circle_distance(self.coords)
        dist_cosine = vectors.calculate_great_circle_distance(
            self.coords, formula='cosine'
        )
        self.assertNotAlmostEqual(dist_cosine, dist_haversine, places=12)
