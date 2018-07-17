# tf-frozen2cc TensorFlow frozen forward model to plain C++ converter

**Experimental** TensorFlow frozen model to plain(STL dependency only) C++ converter.
Only forward(inference) network are supported.

## Development status

Very early phase. Proof of concept, experimental phase.
Only simple MNIST works at the moment.

## Dependencies

These are required for converter script.

* Python 3.5+
* TensorFlow `r1.8`

## How it works

`tf2cc` generates C++ code from JSON description of protobuf frozen model.

## Pros

You can embed trained model into your C++ application without TensorFlow related dependencies(e.g. protobuf, TensorFlow, etc).

## Cons

Not fast. CPU only for now.

## How to convert

Prepare your own frozen model. https://www.tensorflow.org/mobile/prepare_models
Binary frozen file(protobuf) is supported.

```
$ python tf-frozen2cc.py 
```

C++ code of frozen model will be generated.
Then you can add and link generated C++ code to your C++ application.

## API

Please see `examples/mnist` for details.

### Buffer class

`Buffer` class holds memory and weight values of the network.

### NetworkInit

Initialize network. Setup memories for `Buffer`, decode constant(weight) values, etc.

### NetworkForwardExecute

`NetworkForwardExecute` executes network.

`NetworkInit` must be called before calling `NetworkForwardExecute`.

## TODO

* [ ] Support more layers(ops)
* [ ] CIFER10 
* [ ] PRNet https://github.com/lighttransport/prnet-infer

# License

MIT License.

## Third party licecnses

* base64-encdec.inc : Copyright (C) 2004-2008 René Nyffenegger. zlib license. https://github.com/ReneNyffenegger/cpp-base64
* stb_image_write : Public domain.
* json.hpp(used in unit testing) : Copyright © 2013-2018 Niels Lohmann. MIT license. https://github.com/nlohmann/json
* Acutest(unit testing library) : Copyright (c) 2013-2017 Martin Mitas. MIT license. http://github.com/mity/acutest

