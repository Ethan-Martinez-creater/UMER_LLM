# Smoke prompt examples

## pheme example 1 (event 499397048878501888, cutoff SOURCE_ONLY)

```text
SOURCE CLAIM
When white folks start discussing how POC should respond to blatant or covert racism &amp; oppression. -_- #Ferguson http://t.co/bjqZDZFOCY

CURRENT SNAPSHOT
elapsed_time = SOURCE_ONLY
observed_replies = 0
max_depth = 0

SELECTED SOCIAL EVIDENCE
(none)

TASK
Using only the source claim and the social evidence shown above,
classify the source claim as RUMOR or NON_RUMOR.
Return exactly one label.
```

## pheme example 2 (event 499397048878501888, cutoff 15)

```text
SOURCE CLAIM
When white folks start discussing how POC should respond to blatant or covert racism &amp; oppression. -_- #Ferguson http://t.co/bjqZDZFOCY

CURRENT SNAPSHOT
elapsed_time = 15m
observed_replies = 0
max_depth = 0

SELECTED SOCIAL EVIDENCE
(none)

TASK
Using only the source claim and the social evidence shown above,
classify the source claim as RUMOR or NON_RUMOR.
Return exactly one label.
```

## maweibo example 1 (event 3475831861399909, cutoff SOURCE_ONLY)

```text
源帖
如果本届伦敦奥运会结束后，中国代表队能在金牌榜上排第一，现决定：参与转发本微博并关注@慧思外语教育网，每人送一台iPhone4S+iPad3。@5个好友即送iPhone4S，@10个好友以上即送iPhone4S+iPad3！绝对真实！见图上的iPhone4S、iPad3（不玩文字游戏，绝对每人送一台！欢迎截图！叫上你的好友齐参与！）

当前传播状态
elapsed_time = SOURCE_ONLY
observed_replies = 0
max_depth = 0

选中的社会证据
(none)

任务
仅依据上面的源帖和社会证据，
判断该源帖为 RUMOR 或 NON_RUMOR。
只返回一个标签。
```

## maweibo example 2 (event 3475831861399909, cutoff 15)

```text
源帖
如果本届伦敦奥运会结束后，中国代表队能在金牌榜上排第一，现决定：参与转发本微博并关注@慧思外语教育网，每人送一台iPhone4S+iPad3。@5个好友即送iPhone4S，@10个好友以上即送iPhone4S+iPad3！绝对真实！见图上的iPhone4S、iPad3（不玩文字游戏，绝对每人送一台！欢迎截图！叫上你的好友齐参与！）

当前传播状态
elapsed_time = 15m
observed_replies = 6
max_depth = 1

选中的社会证据
[E1]
time = 580s
depth = 1
parent:
如果本届伦敦奥运会结束后，中国代表队能在金牌榜上排第一，现决定：参与转发本微博并关注@慧思外语教育网，每人送一台iPhone4S+iPad3。@5个好友即送iPhone4S，@10个好友以上即送iPhone4S+iPad3！绝对真实！见图上的iPhone4S、iPad3（不玩文字游戏，绝对每人送一台！欢迎截图！叫上你的好友齐参与！）
reply:
希望中国再次创造奇迹[哈哈]@茶煲琳 @liangxianqiang @特級魚腩_風乾的豬屎 @HOHO爹D @SilkH @4eye_s_JY @DIY设计我的家 @IC周倩 @明星经纪吴名 @歐美版堅哥

[E2]
time = 839s
depth = 1
parent:
如果本届伦敦奥运会结束后，中国代表队能在金牌榜上排第一，现决定：参与转发本微博并关注@慧思外语教育网，每人送一台iPhone4S+iPad3。@5个好友即送iPhone4S，@10个好友以上即送iPhone4S+iPad3！绝对真实！见图上的iPhone4S、iPad3（不玩文字游戏，绝对每人送一台！欢迎截图！叫上你的好友齐参与！）
reply:
@等喵的兔子荻 @哆啦2BI梦 @Tanna小僑 @半枝莲花一生境遇 @iris虹笔全身痛 @被周笔畅嫌弃的天使 @艾凉 @angelBI @Sophie_秀 @笨笨爱笔 [加油][冠军诞生]必须兑现啊。。不然。。。。。

任务
仅依据上面的源帖和社会证据，
判断该源帖为 RUMOR 或 NON_RUMOR。
只返回一个标签。
```
