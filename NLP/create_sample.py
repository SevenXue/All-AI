# with open('sample40.sql', 'a+') as f:
#     for i in range(10):
#         sql = "SELECT sms_content FROM sms_000" + str(i) + " WHERE date(send_time)>'2017-10-25' AND date(send_time)<'2017-11-04' union all" + '\n'
#         f.write(sql)

with open('sample40.sql', 'a+') as f:
    for i in range(40, 100):
        sql = "SELECT sms_content FROM sms_00" + str(i) + " WHERE date(send_time)>'2017-10-25' AND date(send_time)<'2017-11-04' union all" + '\n'
        f.write(sql)