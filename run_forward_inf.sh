#! /usr/bin/env bash
python gen_forward.py --alpha .95 --force_cpu waveglow
# python gen_forward.py --alpha .95 --force_cpu --input_text ' Jus lea váigadis dahje guhkesáigásaš buozalvas , de sáhttá ohcat eará oajuid . ' griffinlim
# python gen_forward.py --alpha .95 --force_cpu --input_text 'Jus mánnái lea guvhli šaddan , de ii dárbbaš eará go bossut dan .' griffinlim
# python gen_forward.py --alpha .95 --force_cpu --input_text 'Háviid gal berre bassat , ja limškku giessat birra .' griffinlim
# python gen_forward.py --alpha .95 --force_cpu --input_text 'Jus lea čiekŋalis hávvi , de sáhttá šaddat goaruhit .' griffinlim
# python gen_forward.py --alpha .95 --force_cpu --input_text 'Goaruheami oktavuođas sáhttá sihtat gáldnadandálkasa , muhto dan ii leat álo dárbbašlaš ' griffinlim
# python gen_forward.py --alpha .95 --force_cpu --input_text 'Máŋgii sáhttá leat ávki fátmásis váldit máná . Liegga ja oadjebas fátmmis mánná beassá ráfut .' griffinlim

# python gen_forward.py --alpha .95 --force_cpu --input_text 'Bealjit sáhttet jađgŋot jus gássi dievvá beljiide , dahje maid go šordo girdis čohkkádettiin .' griffinlim

# --alpha value controls the speed

# Jus lea váigadis dahje guhkesáigásaš buozalvas , de sáhttá ohcat eará oajuid .
# Norggas sáhttá oažžut veahki NAV:s .
# Jus mánnái lea guvhli šaddan , de ii dárbbaš eará go bossut dan .
# Háviid gal berre bassat , ja limškku giessat birra .
# Jus lea čiekŋalis hávvi , de sáhttá šaddat goaruhit .
# Goaruheami oktavuođas sáhttá sihtat gáldnadandálkasa , muhto dan ii leat álo dárbbašlaš .
# Máŋgii sáhttá leat ávki fátmásis váldit máná . Liegga ja oadjebas fátmmis mánná beassá ráfut .
# Go bealjit orrot dievvan , de dadjet ahte bealjit leat jađgŋon .
# Bealjit sáhttet jađgŋot jus gássi dievvá beljiide , dahje maid go šordo girdis čohkkádettiin .
# Dearvvašvuohta njálmmi siste lea maid mávssolaš .
# Bániid ferte geallat eastadan dihte ráiggiid .
# Muhto njálmmis leat maid eará hástalusat .

# python waveglow/inference.py -f model_outputs/ -w waveglow/waveglow_14000_st -o audios/ 