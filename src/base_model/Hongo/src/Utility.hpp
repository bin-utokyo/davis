//
//  Utility.hpp
//  Hongo_workspace
//
//  Created by 小川大智 on 2021/10/18.
//

#ifndef Utility_hpp
#define Utility_hpp

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace Hongo {
extern int rectPlaneNum;  // 平面直角座標の番号

class Utility {
 public:
  static double getNormalDist() {
    std::random_device seed_gen;
    // std::mt19937_64 engine(seed_gen());
    std::mt19937_64 engine(0);
    std::normal_distribution<> dist(0.0, 1.0);
    return dist(engine);
  }
  // 0-1間の一様乱数生成
  static double getRandRange01() {
    // 乱数生成器
    static std::mt19937_64 mt64(0);
    // 一様分布生成器
    std::uniform_real_distribution<double> get_rand_uni_real(0.0, 1.0);
    double rand = get_rand_uni_real(mt64);
    return rand;
  }

  static double stNormCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2)));
  }

  static double hubeny(double oLon, double oLat, double dLon, double dLat) {
    double dx = degreeDecimal2Rad(dLon - oLon);
    double dy = degreeDecimal2Rad(dLat - oLat);
    double p = degreeDecimal2Rad((oLat + dLat) / 2);

    const double rx = 6378137.000;                        // 赤道半径(m)
    const double ry = 6356752.314245;                     // 極半径(m)
    const double e = std::sqrt(1 - (ry / rx * ry / rx));  // 離心率
    const double w = std::sqrt(1 - e * e * std::sin(p) * std::sin(p));
    const double m = rx * (1 - e * e) / (w * w * w);
    const double n = rx / w;

    return std::sqrt(dy * m * dy * m +
                     dx * n * std::cos(p) * dx * n * std::cos(p));
  }

  static double euclidean(double ox, double oy, double dx, double dy) {
    double deltax = dx - ox;
    double deltay = dy - oy;
    return std::sqrt(deltax * deltax + deltay * deltay);
  }

  static std::pair<double, double> xy2LonLat(double x, double y) {
    // numは系番号
    using std::pow;
    std::unordered_map<int, std::pair<double, double>>
        originalPoints;  //(経度、緯度）
    originalPoints[1] =
        std::make_pair(degree2Rad(129, 30, 0), degree2Rad(33, 0, 0));
    originalPoints[2] =
        std::make_pair(degree2Rad(131, 0, 0), degree2Rad(33, 0, 0));
    originalPoints[3] =
        std::make_pair(degree2Rad(132, 10, 0), degree2Rad(36, 0, 0));
    originalPoints[4] =
        std::make_pair(degree2Rad(133, 30, 0), degree2Rad(33, 0, 0));
    originalPoints[5] =
        std::make_pair(degree2Rad(134, 20, 0), degree2Rad(36, 0, 0));
    originalPoints[6] =
        std::make_pair(degree2Rad(136, 0, 0), degree2Rad(36, 0, 0));
    originalPoints[7] =
        std::make_pair(degree2Rad(137, 10, 0), degree2Rad(36, 0, 0));
    originalPoints[8] =
        std::make_pair(degree2Rad(138, 30, 0), degree2Rad(36, 0, 0));
    originalPoints[9] =
        std::make_pair(degree2Rad(139, 50, 0), degree2Rad(36, 0, 0));
    originalPoints[10] =
        std::make_pair(degree2Rad(140, 50, 0), degree2Rad(40, 0, 0));
    originalPoints[11] =
        std::make_pair(degree2Rad(140, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[12] =
        std::make_pair(degree2Rad(142, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[13] =
        std::make_pair(degree2Rad(144, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[14] =
        std::make_pair(degree2Rad(142, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[15] =
        std::make_pair(degree2Rad(127, 30, 0), degree2Rad(26, 0, 0));
    originalPoints[16] =
        std::make_pair(degree2Rad(124, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[17] =
        std::make_pair(degree2Rad(131, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[18] =
        std::make_pair(degree2Rad(136, 0, 0), degree2Rad(20, 0, 0));
    originalPoints[19] =
        std::make_pair(degree2Rad(154, 0, 0), degree2Rad(26, 0, 0));

    const double lambda0 = originalPoints.at(rectPlaneNum).first;
    const double phy0 = originalPoints.at(rectPlaneNum).second;
    const double a = 6378137.000;    // 赤道半径(m)
    const double F = 298.257222101;  // 逆扁平率
    const double m0 = 0.9999;        // x軸上での縮尺係数

    const double n = 1 / (2 * F - 1);
    std::vector<double> A(6), beta(6), delta(7);
    A.at(0) = 1. + pow(n, 2) / 4. + pow(n, 4) / 64.;
    A.at(1) = -3. / 2. * (n - pow(n, 3) / 8. - pow(n, 5) / 64.);
    A.at(2) = 15. / 16. * (pow(n, 2) - pow(n, 4) / 4.);
    A.at(3) = -35. / 48. * (pow(n, 3) - 5. / 16. * pow(n, 5));
    A.at(4) = 315. / 512. * pow(n, 4);
    A.at(5) = -693. / 1280. * pow(n, 5);
    beta.at(1) = 1. / 2. * n - 2. / 3. * pow(n, 2) + 37. / 96. * pow(n, 3) -
                 pow(n, 4) / 360. - 81. / 512. * pow(n, 5);
    beta.at(2) = pow(n, 2) / 48. + pow(n, 3) / 15. - 437. / 1440. * pow(n, 4) +
                 46. / 105. * pow(n, 5);
    beta.at(3) = 17. / 480. * pow(n, 3) - 37. / 840. * pow(n, 4) -
                 209. / 4480. * pow(n, 5);
    beta.at(4) = 4397. / 161280. * pow(n, 4) - 11. / 504. * pow(n, 5);
    beta.at(5) = 4583. / 161280. * pow(n, 5);
    delta.at(1) = 2. * n - 2. / 3. * pow(n, 2) - 2. * pow(n, 3) +
                  116. / 45. * pow(n, 4) + 26. / 45. * pow(n, 5) -
                  2854. / 675. * pow(n, 6);
    delta.at(2) = 7. / 3. * pow(n, 2) - 8. / 5. * pow(n, 3) -
                  227. / 45. * pow(n, 4) + 2704. / 315. * pow(n, 5) +
                  2323. / 945. * pow(n, 6);
    delta.at(3) = 56. / 15. * pow(n, 3) - 136. / 35. * pow(n, 4) -
                  1262. / 105. * pow(n, 5) + 73814. / 2835. * pow(n, 6);
    delta.at(4) = 4279. / 630. * pow(n, 4) - 332. / 35. * pow(n, 5) -
                  399572. / 14175. * pow(n, 6);
    delta.at(5) = 4174. / 315. * pow(n, 5) - 144838. / 6237. * pow(n, 6);
    delta.at(6) = 601676. / 22275. * pow(n, 6);

    double S = A.at(0) * phy0;
    for (std::size_t i = 1; i <= 5; i++) {
      S += A.at(i) * std::sin(2 * i * phy0);
    }
    S *= m0 * a / (1 + n);

    const double Abar = m0 * a / (1 + n) * A.at(0);

    const double ksi = (x + S) / Abar;
    const double eta = y / Abar;

    double ksidot = ksi;
    double etadot = eta;
    for (int i = 1; i <= 5; i++) {
      ksidot -= beta.at(i) * std::sin(2 * i * ksi) * std::cosh(2 * i * eta);
      etadot -= beta.at(i) * std::cos(2 * i * ksi) * std::sinh(2 * i * eta);
    }

    const double khi = std::asin(std::sin(ksidot) / std::cosh(etadot));
    double lambda = lambda0 + std::atan(std::sinh(etadot) / std::cos(ksidot));
    double phy = khi;
    for (int i = 1; i <= 6; i++) {
      phy += delta.at(i) * std::sin(2 * i * khi);
    }
    return std::make_pair(rad2DegreeDecimal(lambda), rad2DegreeDecimal(phy));
  }

  static std::pair<double, double> lonLat2xy(double lon, double lat) {
    using std::pow;
    std::unordered_map<int, std::pair<double, double>>
        originalPoints;  //(経度、緯度）
    originalPoints[1] =
        std::make_pair(degree2Rad(129, 30, 0), degree2Rad(33, 0, 0));
    originalPoints[2] =
        std::make_pair(degree2Rad(131, 0, 0), degree2Rad(33, 0, 0));
    originalPoints[3] =
        std::make_pair(degree2Rad(132, 10, 0), degree2Rad(36, 0, 0));
    originalPoints[4] =
        std::make_pair(degree2Rad(133, 30, 0), degree2Rad(33, 0, 0));
    originalPoints[5] =
        std::make_pair(degree2Rad(134, 20, 0), degree2Rad(36, 0, 0));
    originalPoints[6] =
        std::make_pair(degree2Rad(136, 0, 0), degree2Rad(36, 0, 0));
    originalPoints[7] =
        std::make_pair(degree2Rad(137, 10, 0), degree2Rad(36, 0, 0));
    originalPoints[8] =
        std::make_pair(degree2Rad(138, 30, 0), degree2Rad(36, 0, 0));
    originalPoints[9] =
        std::make_pair(degree2Rad(139, 50, 0), degree2Rad(36, 0, 0));
    originalPoints[10] =
        std::make_pair(degree2Rad(140, 50, 0), degree2Rad(40, 0, 0));
    originalPoints[11] =
        std::make_pair(degree2Rad(140, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[12] =
        std::make_pair(degree2Rad(142, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[13] =
        std::make_pair(degree2Rad(144, 15, 0), degree2Rad(44, 0, 0));
    originalPoints[14] =
        std::make_pair(degree2Rad(142, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[15] =
        std::make_pair(degree2Rad(127, 30, 0), degree2Rad(26, 0, 0));
    originalPoints[16] =
        std::make_pair(degree2Rad(124, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[17] =
        std::make_pair(degree2Rad(131, 0, 0), degree2Rad(26, 0, 0));
    originalPoints[18] =
        std::make_pair(degree2Rad(136, 0, 0), degree2Rad(20, 0, 0));
    originalPoints[19] =
        std::make_pair(degree2Rad(154, 0, 0), degree2Rad(26, 0, 0));

    const double lambda = degreeDecimal2Rad(lon);
    const double phy = degreeDecimal2Rad(lat);

    const double lambda0 = originalPoints.at(rectPlaneNum).first;
    const double phy0 = originalPoints.at(rectPlaneNum).second;
    const double a = 6378137.000;    // 赤道半径(m)
    const double F = 298.257222101;  // 逆扁平率
    const double m0 = 0.9999;        // x軸上での縮尺係数

    const double n = 1 / (2 * F - 1);

    std::vector<double> A(6), alpha(6);
    A.at(0) = 1. + pow(n, 2) / 4. + pow(n, 4) / 64.;
    A.at(1) = -3. / 2. * (n - pow(n, 3) / 8. - pow(n, 5) / 64.);
    A.at(2) = 15. / 16. * (pow(n, 2) - pow(n, 4) / 4.);
    A.at(3) = -35. / 48. * (pow(n, 3) - 5. / 16. * pow(n, 5));
    A.at(4) = 315. / 512. * pow(n, 4);
    A.at(5) = -693. / 1280. * pow(n, 5);
    alpha.at(1) = n / 2. - 2. / 3. * pow(n, 2) + 5. / 16. * pow(n, 3) +
                  41. / 180. * pow(n, 4) - 127. / 288. * pow(n, 5);
    alpha.at(2) = 13. / 48. * pow(n, 2) - 3. / 5. * pow(n, 3) +
                  557. / 1440. * pow(n, 4) + 281. / 630. * pow(n, 5);
    alpha.at(3) = 61. / 240. * pow(n, 3) - 103. / 140. * pow(n, 4) +
                  15061. / 26880. * pow(n, 5);
    alpha.at(4) = 49561. / 161280. * pow(n, 4) - 179. / 168. * pow(n, 5);
    alpha.at(5) = 34729. / 80640. * pow(n, 5);

    double S = A.at(0) * phy0;
    for (int i = 1; i <= 5; i++) {
      S += A.at(i) * std::sin(2 * i * phy0);
    }
    S *= m0 * a / (1 + n);

    const double Abar = m0 * a / (1 + n) * A.at(0);

    const double lambdac = std::cos(lambda - lambda0);
    const double lambdas = std::sin(lambda - lambda0);

    const double t =
        std::sinh(std::atanh(std::sin(phy)) -
                  2 * std::sqrt(n) / (1 + n) *
                      std::atanh(2 * std::sqrt(n) / (1 + n) * std::sin(phy)));
    const double tdot = std::sqrt(1 + t * t);

    const double ksidot = std::atan(t / lambdac);
    const double etadot = std::atanh(lambdas / tdot);

    double x = ksidot;
    double y = etadot;
    for (int i = 1; i <= 5; i++) {
      x += alpha.at(i) * std::sin(2 * i * ksidot) * std::cosh(2 * i * etadot);
      y += alpha.at(i) * std::cos(2 * i * ksidot) * std::sinh(2 * i * etadot);
    }
    x = Abar * x - S;
    y = Abar * y;
    return std::make_pair(x, y);
  }

  static double degree2Rad(double d, double m, double s) {
    return (d + m / 60.0 + s / 3600.0) / 180.0 * M_PI;
  }
  static double degreeDecimal2Rad(double degree) {
    return degree / 180.0 * M_PI;
  }
  static double rad2DegreeDecimal(double rad) { return rad / M_PI * 180.0; }

  static double direction(double oX, double oY, double dX, double dY) {
    // 0~2*pi
    double dx = dX - oX;
    double dy = dY - oY;
    if (dx > 0) {
      if (dy >= 0) {
        return std::atan(dy / dx);
      } else {
        return 2 * M_PI - std::atan(-dy / dx);
      }
    } else if (dx < 0) {
      if (dy >= 0) {
        return M_PI - std::atan(-dy / dx);
      } else {
        return M_PI + std::atan(dy / dx);
      }
    } else {
      if (dy > 0) {
        return M_PI / 2;
      } else if (dy < 0) {
        return M_PI * 3 / 2;
      } else {
        return 0.0;
      }
    }
  }
  static double oppositeDirection(double direction) {
    if (direction < M_PI) {
      return direction + M_PI;
    } else {
      return direction - M_PI;
    }
  }
  static double angle(double d1, double d2) {
    if (d1 > d2) {
      std::swap(d1, d2);
    }
    double theta = d2 - d1;
    if (theta > M_PI) {
      theta = 2 * M_PI - theta;
    }
    return theta;
  }

  std::vector<std::string> split(std::string &input, char delimiter) {
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter)) {
      result.push_back(field);
    }
    return result;
  }
};
}  // namespace Hongo

#endif /* Utility_hpp */
