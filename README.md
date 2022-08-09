# miptAI-well-pads
Моё решение задачи <<Бурение нефтяных скважин>>, предложенной на форуме МФТИ по математике и ИИ весной 2021 года.

# Описание задачи

Ваша цель – провести размещение мест бурения группы нефтяных скважин (куста) с учетом карты разрешенных областей оптимальным с точки зрения стоимости образом.

Добыча нефти – сложный технологический процесс. А подготовка к добыче еще сложнее. Трехфазная смесь нефти, газа и воды (флюид) находится в порах проводящих пород (для лучшего понимания можно представить каменную губку) под большим давлением. Если пробурить скважину к области залегания флюида, то под действием энергии пласта (давления) флюид начнет поступать в забой скважины, а по скважине – к устью на поверхность. Со временем давление в пласте снижается, поэтому используют насосы (газлифты) и производят геолого-технологические мероприятия для поддержания уровня добычи.

Скважины бывают как вертикальные (наклонно-направленные), так и горизонтальные, а в последнее время активно используются многозабойные скважины. Длина скважины может достигать нескольких километров. Бурение скважины – дорогой процесс, поэтому добывающие компании стремятся пробурить скважины меньшей длины в рамках технологических ограничений, но достаточной для того, чтобы достичь точки залегания флюида (геологической цели).

Необходимо собирать и транспортировать флюид, поступающий из устья скважины. Для этого строят наземную сеть трубопроводов, по которым углеводороды из с устья каждой скважины собираются в магистральный трубопровод и транспортируются к местам переработки. Чтобы сэкономить на наземном обустройстве месторождения, устья скважин объединяют в группы (кустовые площадки). В каждой кустовой площадке скважины располагаются на расстоянии нескольких метров друг от друга. Выбор места кустовой площадки – ответственный процесс, поскольку требует учета наличие природных объектов (лесов, болот, рек, озер и др.), которые не позволят расположить кустовую площадку вблизи них или влияют на стоимость размещения площадки.

Необходимо провести кустование с учетом карты стоимостей: распределить геологические цели по кустовым площадкам оптимальным по стоимости образом. Соединение кустовых площадок наземной транспортной сетью сбора углеводородов не учитывать. Кустовая площадка имеет квадратную форму. Узлы запретной зоны не должны попадать в границы кустовой площадки. Длина ребра a кустовой площадки пропорциональна количеству скважин $N_m$ на данной площадке $a= \alpha N_m$. Координатами кустовой площадки является координаты ее геометрического центра. Стоимость размещения кустовой площадки в разных зонах – различная. Коэффициент стоимости рассчитывается как среднее арифметическое коэффициентов всех узлов карты стоимостей, попадающих в границы кустовой площадки.

Длина скважины должна быть меньше предельной допустимой длины скважины $l_{max}$. Расстояние между двумя скважинами должно превышать $d_{min}$. Необходимо учитывать, что скважина должны быть введена в эксплуатацию (пробурена) в момент времени $t$. Возможно использование только одной буровой установки для бурения всех скважин. Монтаж буровой установки на кустовой площадке стоит $С_0$. Стоимость передвижения буровой установки в пределах кустовой площадки для бурения скважины – $С_2$. Демонтаж буровой установки с кустовой площадки после окончания бурения – $С_1$. К примеру, если необходимо пробурить три скважины (устья первой и третьей принадлежат первой кустовой площадке, а устье второй скважины принадлежит второй кустовой площадке, причем время ввода в эксплуатацию второй скважины между временем ввода в эксплуатацию первой и третьей скважины), то сначала необходимо пробурить первую скважину, демонтировать буровую установку с первой кустовой площадки, смонтировать буровую установку на второй кустовой площадке, пробурить вторую скважину, демонтировать буровую установку, смонтировать ее на первой кустовой площадке и пробурить третью скважину. Стоимость бурения скважины рассчитывается по формуле:

$C_{w_i} = C_3 l_{w_i}$, где $l_{w_i}$ – длина $i$-той скважины.

Общая стоимость всех скважин рассчитывается по формуле:

$C_w = \Sigma_{i=1}^{N_{w}} k_m C_{w_i}$

Стоимость обустройства кустовой площадки рассчитывается по формуле:

$C_{wp_m} = C_0 + C_1 + C_2 N_m$

Стоимость обустройства всех кустовых площадок рассчитывается по формуле:

$C_{wp} = \Sigma_{i=1}^{N_{wp}} k_m C_{w_i}$

Необходимо минимизировать общую стоимость разработки и обустройства:

$C = C_w + C_{wp}$

Следует учитывать, что в случае превышения предельной длины скважины $l_{max}$ или расположения скважин на расстоянии меньшем $d_{min}$ к друг другу, решение не будет засчитано.

Дополнительно Вам предоставляется `wrapper.py` – скрипт, который можно использовать для оценки стоимости своего решения.
