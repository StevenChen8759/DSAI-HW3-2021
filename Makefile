STUDENT_ID := NE6091051

.PHONY: download upload

init:
	curl https://pyenv.run | bash
	pipenv --python 3.8

run:
	pipenv run python main.py $(ARGS)

train:
	pipenv run python train.py $(ARGS)

ziphw:
	$(eval VER := $(shell cat version.txt))
	$(eval FILE_PREFIX := $(STUDENT_ID)-$(VER))
	@echo "Filename: $(FILE_PREFIX).zip"
	@echo
	@mkdir $(FILE_PREFIX)
	@cp Pipfile* $(FILE_PREFIX)
	@cp *.joblib $(FILE_PREFIX)
	@cp main.py $(FILE_PREFIX)
	@mkdir $(FILE_PREFIX)/predictor
	@cp predictor/*.py $(FILE_PREFIX)/predictor
	@mkdir $(FILE_PREFIX)/utils
	@cp utils/*.py $(FILE_PREFIX)/utils
	zip $(FILE_PREFIX) $(FILE_PREFIX)/*
	zip $(FILE_PREFIX) $(FILE_PREFIX)/utils/*
	zip $(FILE_PREFIX) $(FILE_PREFIX)/predictor/*
	@mv $(FILE_PREFIX).zip upload
	@echo
	ls -al upload
	@rm -r $(FILE_PREFIX)

upload:
	$(eval VER := $(shell cat version.txt))
	$(eval FILE_PREFIX := $(STUDENT_ID)-$(VER))
	@echo -ne '\007'
	printf "put upload/$(FILE_PREFIX).zip" | sftp -P 22 $(STUDENT_ID)@140.116.247.123:upload

download:
	@test ! -d "./download" && mkdir "./download" || cd .
	@echo -ne '\007'
	printf "get -r download/student/$(STUDENT_ID) \r\nget -r download/information" | sftp -P 22 $(STUDENT_ID)@140.116.247.123
	rm -r ./download/*
	mv $(STUDENT_ID) ./download/result
	mv ./information ./download

submit: ziphw upload

feedback: download