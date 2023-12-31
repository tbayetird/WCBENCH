coco_to_dsd={}

amazon_to_coco={
                'person':['Carpenter','Athlete','Astronaut','Attorney','Bartender','Boy','Bride','Bridegroom','Bridesmaid','Child','Cyclist','Dentist','Diver','Doctor','Female','Face','Fireman','Fist','Gardener','Girl','Guitarist','Gymnast','Hand','Head','Human','Kid','Lip','Man','Monk','Musician','Nail','Neck','Ninja','Officer','Patient','Pedestrian','People','Performer','Person','Photographer','Pianist','Priest','Scientist','Senior Citizen','Server','Shoulder','Student','Surgeon','Teacher','Teen','Teeth','Throat','Tourist','Woman'],
                'bicycle':['Bicycle','Tandem Bicycle','Tricycle'],
                'car':['Antique Car','Automobile','Cab','Car','Car Mirror','Car Seat','Car Trunk','Car Wheel','Police Car','Race Car','Sports Car','Streetcar','Suv','Taxi'],
                'motorcycle':['Bike','Motor Scooter','Motorcycle','Scooter'],
                'airplane':['Airplane','Biplane'],
                'bus':['Bus','Minibus','School Bus','Tour Bus'],
                'train':['Bullet Train','Train'],
                'truck':['Fire Truck','Tow Truck','Trailer Truck','Truck'],
                'boat':['Battleship','Boat','Cruise Ship','Sailboat','Ship','Submarine'],
                'traffic_light':['Traffic Light'],
                'fire_hydrant':['Fire Hydrant'],
                'stop_sign':['Stopsign'],
                'parking_meter':['Parking Meter'],
                'bench':['Bench','Park Bench'],
                'bird':['Chicken','Bird','African Grey Parrot','Albatross','Bald Eagle','Ballerina','Bat','Cock Bird','Cockatoo','Crane Bird','Crow','Duck','Eagle','Hawk','Heron','Hummingbird','Kiwi Bird','Ostrich','Owl','Parrot','Pelican','Penguin','Pigeon','Rooster','Seagull','Sparrow','Swan','Turkey Bird'],
                'cat':['Black Cat','Cat','Egyptian Cat'],
                'dog':['Afghan Hound','Bulldog','Chihuahua','Dog','Eskimo Dog','French Bulldog','German Shepherd','Husky','Pitbull','Police Dog','White Dog'],
                'horse':['Horse'],
                'sheep':['Sheep'],
                'cow':['Cow','Dairy Cow'],
                'elephant':['Elephant'],
                'bear':['Bear','Black Bear','Brown Bear','Polar Bear'],
                'zebra':['Zebra'],
                'giraffe':['Giraffe'],
                'backpack':['Backpack'],
                'umbrella':['Garden Umbrella','Patio Umbrella','Umbrella'],
                'handbag':['Handbag'],
                'tie':['Tie','Necktie'],
                'suitcase':['Briefcase','Suitcase'],
                'frisbee':['Frisbee'],
                'skis':[],
                'snowboard':[],
                'sports_ball':['Golf Ball','Soccer Ball','Tennis Ball'],
                'kite':['Kite'],
                'baseball_bat':['Baseball Bat'],
                'baseball_glove':['Baseball Glove'],
                'skateboard':['Skateboard'],
                'surfboard':['Surfboard'],
                'tennis_racket':['Racket','Tennis Racket'],
                'bottle':['Beer Bottle','Bottle','Ink Bottle','Water Bottle','Wine Bottle'],
                'wine_glass':['Wine Glass'],
                'cup':['Beer Glass','Coffee Cup','Cup','Measuring Cup'],
                'fork':['Fork'],
                'knife':['Knife'],
                'spoon':['Spoon','Wooden Spoon'],
                'bowl':['Bowl','Soup Bowl','Mixing Bowl'],
                'banana':['Banana'],
                'apple':['Apple'],
                'sandwich':['Sandwich','Sandwich Wrap'],
                'orange':['Orange'],
                'broccoli':['Broccoli'],
                'carrot':['Carrot'],
                'hot_dog':['Hot Dog'],
                'pizza':['Pizza'],
                'donut':['Donut'],
                'cake':['Wedding Cake','Birthday Cake','Cake'],
                'chair':['Chair'],
                'couch':['Armchair','Couch'],
                'potted_plant':['Potted Plant'],
                'bed':['Bed','Bunk Bed'],
                'dining_table':['Dining Table'],
                'toilet':['Toilet'],
                'tv':['Television','TV'],
                'laptop':['Hand-Held Computer','Laptop'],
                'mouse':['Mouse'],
                'remote':['Remote Control'],
                'keyboard':['Computer Keyboard','Keyboard'],
                'cell_phone':['Cell Phone','Iphone'],
                'microwave':['Microwave'],
                'oven':['Dutch Oven','Oven'],
                'toaster':['Toaster'],
                'sink':['Double Sink','Sink','Sink Faucet'],
                'refrigerator':['Refrigerator'],
                'book':['Book','Diary','Novel'],
                'clock':['Alarm Clock','Analog Clock','Clock','Digital Clock','Digital Watch','Wall Clock'],
                'vase':['Vase'],
                'scissors':['Scissors'],
                'teddy_bear':['Teddy Bear'],
                'hair_drier':['Blow Dryer','Hair Drier'],
                'toothbrush':['Toothbrush']
}

google_to_coco={
                'person':['prince','high priest','cardinal','motorcycle racer','jazz pianist','keyboard player','tennis player','jazz guitarist','shopkeeper','fisherman','bus driver','skier','pianist','bride','soccer player','blue-collar worker','construction worker','golfer','skateboarder','bassist','guitarist','police officer','spokesperson','passenger','soldier','human','bodybuilder','baseball player','people','face','foot','cheek','people on beach','gull','athlete','throat','singer','hand','finger','chest','mouth','chin','nose','forehead','eye','ear','leg','white-collar worker','zookeeper','wildlife biologist','people in nature','firefighter','head','wrist','abdomen','arm','fashion model','shoulder','neck','waist','pedestrian','human body','child','football player','rugby player','player','photograph','military person','human leg','musician','artist','music artist','elbow','knee','thumb'],
                'bicycle':['bicycle','cyclo-cross bicycle','road bicycle','racing bicycle'],
                'car':['cadillac','taxi','police car','race car','formula one car','compact car','performance car','antique car','sports car','stock car racing','world rally car','full-size car','luxury vehicle','car','personal luxury car','mid-size car','classic car','commercial vehicle'],
                'motorcycle':['motorcycle'],
                'airplane':['monoplane','airplane','biplane'],
                'bus':['bus'],
                'train':['train','locomotive','bullet train','railroad car'],
                'truck':['truck','trailer truck','monster truck','pickup truck'],
                'boat':['cargo ship','royal yacht','sailboat','ship','naval ship','warship','passenger ship','steamboat'],
                'traffic_light':['traffic light'],
                'fire_hydrant':['fire hydrant'],
                'stop_sign':['stop sign'],
                'parking_meter':['parking meter'],
                'bench':['outdoor bench','bench'],
                'bird':['pileated woodpecker','ivory-billed woodpecker','yellow warbler','red breasted nuthatch','condor','black capped chickadee','chickadee','field sparrow','canada goose','barn owl','goose','green heron','bald eagle','sea eagle','red headed woodpecker',    'atlantic puffin','puffin','belted kingfisher','coraciiformes','lovebird','waterfowl','ciconiiformes','pelecaniformes','phasianidae','galliformes','buzzard','woodpecker','laughing kookaburra','lorikeet','accipitriformes','old world flycatcher','rallidae','toucan','american mourning dove','ostrich','penguin','great egret','mouse eared bat','swan','black swan','american crow','raven','hawk','great heron','great blue heron','pigeons and doves','hummingbird','egret','heron','duck','seaduck','eagle','sparrow','bat','shorebird','seabird','geese and swans','ducks','bird of prey','owl','bird','water bird','common gallinule','crane-like bird','songbird','perching bird','crow-like bird','fish crow','crow','new caledonian crow','parrot','falconiformes'],
                'cat':['cat','small to medium-sized cats','domestic short-haired cat','big cats'],
                'dog':['king shepherd','gun dog','water dog','molosser','beaglier','bulldog','german shepherd dog','giant dog breed','herding dog','guard dog','street dog','french bulldog','boston terrier','dog','dog breed','companion dog','hunting dog','ancient dog breeds'],
                'horse':['horse','mustang horse','stallion'],
                'sheep':['sheep','barbary sheep'],
                'cow':[],
                'elephant':['indian elephant','elephant','african elephant'],
                'bear':['american black bear','bear','sloth bear','sun bear','spectacled bear'],
                'zebra':['zebra'],
                'giraffe':['giraffe','giraffidae'],
                'backpack':['backpack'],
                'umbrella':['umbrella'],
                'handbag':[],
                'tie':['tie'],
                'suitcase':[],
                'frisbee':[],
                'skis':['ski'],
                'snowboard':[],
                'sports_ball':['ball','tennis ball','soccer ball'],
                'kite':['kite'],
                'baseball_bat':['baseball bat'],
                'baseball_glove':['baseball glove'],
                'skateboard':[],
                'surfboard':['surfboard'],
                'tennis_racket':['racket','tennis racket'],
                'bottle':['bottle','glass bottle','wine bottle','beer bottle','water bottle','plastic bottle','bottled water'],
                'wine_glass':['wine glass','highball glass'],
                'cup':['martini glass','beer glass','pint glass','teacup','coffee cup','cup','highball glass'],
                'fork':['fork'],
                'knife':[],
                'spoon':['spoon'],
                'bowl':['bowl','mixing bowl'],
                'banana':['banana'],
                'apple':['apple'],
                'sandwich':['sandwich'],
                'orange':['bitter orange','valencia orange','orange'],
                'broccoli':[],
                'carrot':['carrot'],
                'hot_dog':['hot dog','hot dog bun','coney island hot dog'],
                'pizza':['pizza','california-style pizza'],
                'donut':[],
                'cake':['birthday cake','sugar cake','cake','sugar cake'],
                'chair':['chair'],
                'couch':['couch','studio couch'],
                'potted_plant':['flowerpot'],
                'bed':['bedroom','bed','bedding'],
                'dining_table':[],
                'toilet':[],
                'tv':['television'],
                'laptop':['laptop','touchpad'],
                'mouse':[],
                'remote':[],
                'keyboard':['computer keyboard'],
                'cell_phone':['mobile phone'],
                'microwave':['microwave oven'],
                'oven':[],
                'toaster':[],
                'sink':['sink','bathroom sink','kitchen sink','tap'],
                'refrigerator':['refrigerator','freezer'],
                'book':['book','comic book'],
                'clock':['clock','watch','analog watch','wall clock','quartz clock','longcase clock'],
                'vase':['vase'],
                'scissors':[],
                'teddy_bear':['teddy bear'],
                'hair_drier':[],
                'toothbrush':[]
}




microsoft_to_coco={
                'person':['person','man','people','family','male','crowd','player','rider','woman','ear','hair','lady','girl','child','boy','pedestrians','hand','female'],
                'bicycle':['bicycle','tricycle'],
                'car':['car'],
                'motorcycle':['scooter','motorcycle'],
                'airplane':['plane','airplane'],
                'bus':['bus'],
                'train':['train'],
                'truck':['truck'],
                'boat':['boat','ship'],
                'traffic_light':[],
                'fire_hydrant':[],
                'stop_sign':['stop'],
                'parking_meter':[],
                'bench':['bench'],
                'bird':['bird','aquatic bird','gallinule','parrot','gull','bird of prey','eagle','duck','crane','gallinaceous bird','chicken','rooster','hawk','hummingbird','swan','goose','owl','pigeon','heron','penguin','ostrich','ibis',],
                'cat':['big cat','cat','domestic cat','wildcat'],
                'dog':['dog','wild dog'],
                'horse':['horse'],
                'sheep':['sheep'],
                'cow':['cow'],
                'elephant':['elephant'],
                'bear':['bear'],
                'zebra':['zebra'],
                'giraffe':['giraffe'],
                'backpack':[],
                'umbrella':['umbrella'],
                'handbag':[],
                'tie':['necktie'],
                'suitcase':[],
                'frisbee':[],
                'skis':[],
                'snowboard':[],
                'sports_ball':['ball'],
                'kite':['kite'],
                'baseball_bat':[],
                'baseball_glove':[],
                'skateboard':[],
                'surfboard':[],
                'tennis_racket':['tennis','racket'],
                'bottle':['bottle'],
                'wine_glass':[],
                'cup':['glass','cup','coffee cup'],
                'fork':['fork'],
                'knife':[],
                'spoon':[],
                'bowl':['bowl'],
                'banana':['banana'],
                'apple':['custard apple'],
                'sandwich':['sandwich','sandwich plate'],
                'orange':['orange'],
                'broccoli':[],
                'carrot':['carrot'],
                'hot_dog':[],
                'pizza':['pizza'],
                'donut':[],
                'cake':['cake'],
                'chair':['chair'],
                'couch':[],
                'potted_plant':[],
                'bed':['bed','baby bed','bedroom'],
                'dining_table':['dining table'],
                'toilet':['toilet'],
                'tv':['television'],
                'laptop':[],
                'mouse':[],
                'remote':['remote'],
                'keyboard':[],
                'cell_phone':[],
                'microwave':[],
                'oven':['oven'],
                'toaster':[],
                'sink':['sink'],
                'refrigerator':['refrigerator'],
                'book':['book'],
                'clock':['watch','clock'],
                'vase':[],
                'scissors':[],
                'teddy_bear':[],
                'hair_drier':[],
                'toothbrush':[]
}

gyfcc_to_coco={
                'person':['human','kid','man','mechanical_man'],
                'bicycle':['bike'],
                'car':['ambulance','car','motorcar','passenger_vehicle','pickup_truck','suv','sport_car','racing_car','stock_car','squad_car'],
                'motorcycle':['sidecar','scooter'],
                'airplane':['aircraft','plane','jet-propelled_plane','attack_aircraft','hydroplane'],
                'bus':['minibus'],
                'train':['trolley_car','railroad_train'],
                'truck':['truck','fire_truck'],
                'boat':['boat','ship','ferryboat','pirate_ship','powerboat','watercraft','sailing_ship','fireboat','u-boat','combat_ship','speedboat','attack_aircraft_carrier','lifeboat'],
                'traffic_light':[],
                'fire_hydrant':['hydrant'],
                'stop_sign':[],
                'parking_meter':[],
                'bench':['bench'],
                'bird':[],
                'cat':[],
                'dog':[],
                'horse':['pony'],
                'sheep':['sheep'],
                'cow':[],
                'elephant':['elephant'],
                'bear':[],
                'zebra':[],
                'giraffe':['giraffa_camelopardalis'],
                'backpack':[],
                'umbrella':['umbrella'],
                'handbag':[],
                'tie':['black_tie','tie'],
                'suitcase':[],
                'frisbee':[],
                'skis':['ski'],
                'snowboard':['snowboard'],
                'sports_ball':['ball'],
                'kite':[],
                'baseball_bat':[],
                'baseball_glove':[],
                'skateboard':['skate','skateboard'],
                'surfboard':[],
                'tennis_racket':[],
                'bottle':['bottle'],
                'wine_glass':['wineglass'],
                'cup':['cup','mug'],
                'fork':[],
                'knife':['knife'],
                'spoon':['spoon'],
                'bowl':['bowl'],
                'banana':['banana'],
                'apple':['apple'],
                'sandwich':['sandwich'],
                'orange':['orange'],
                'broccoli':['broccoli'],
                'carrot':[],
                'hot_dog':[],
                'pizza':['pizza_pie'],
                'donut':[],
                'cake':[],
                'chair':['chair'],
                'couch':['couch'],
                'potted_plant':['flowerpot'],
                'bed':['bed','bedchamber'],
                'dining_table':[],
                'toilet':[],
                'tv':['television_system'],
                'laptop':['laptop_computer','notebook_computer'],
                'mouse':['computer_mouse'],
                'remote':['remote'],
                'keyboard':['keyboard'],
                'cell_phone':['phone'],
                'microwave':[],
                'oven':['stove','oven','cooking_stove'],
                'toaster':['toaster'],
                'sink':[],
                'refrigerator':['fridge'],
                'book':['book'],
                'clock':['clock'],
                'vase':['vase'],
                'scissors':[],
                'teddy_bear':['teddy_bear'],
                'hair_drier':[],
                'toothbrush':[]
}
