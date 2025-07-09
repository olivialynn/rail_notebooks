Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f46b66809a0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.009381  0.009333  
    1      25.391064  0.106074  0.092584  
    2      24.304707  0.074859  0.046678  
    3      25.291103  0.084409  0.080464  
    4      25.096743  0.070115  0.038200  
    ...          ...       ...       ...  
    99995  24.737946  0.109835  0.094265  
    99996  24.224169  0.191820  0.109653  
    99997  25.613836  0.198935  0.121044  
    99998  25.274899  0.048308  0.030128  
    99999  25.699642  0.105305  0.094104  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.589797</td>
          <td>0.404105</td>
          <td>26.546053</td>
          <td>0.144092</td>
          <td>25.938151</td>
          <td>0.074654</td>
          <td>25.186022</td>
          <td>0.062616</td>
          <td>24.683381</td>
          <td>0.076782</td>
          <td>24.147592</td>
          <td>0.107553</td>
          <td>0.009381</td>
          <td>0.009333</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.030870</td>
          <td>0.561046</td>
          <td>27.559673</td>
          <td>0.334134</td>
          <td>26.662932</td>
          <td>0.140703</td>
          <td>26.342053</td>
          <td>0.171725</td>
          <td>25.878430</td>
          <td>0.215464</td>
          <td>25.172172</td>
          <td>0.256963</td>
          <td>0.106074</td>
          <td>0.092584</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.492169</td>
          <td>0.771073</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.844090</td>
          <td>0.373119</td>
          <td>26.210370</td>
          <td>0.153464</td>
          <td>25.051141</td>
          <td>0.106087</td>
          <td>24.409861</td>
          <td>0.135077</td>
          <td>0.074859</td>
          <td>0.046678</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.542909</td>
          <td>0.797151</td>
          <td>32.504728</td>
          <td>3.867158</td>
          <td>27.299885</td>
          <td>0.240997</td>
          <td>26.300818</td>
          <td>0.165801</td>
          <td>25.442879</td>
          <td>0.148981</td>
          <td>25.230786</td>
          <td>0.269569</td>
          <td>0.084409</td>
          <td>0.080464</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.214182</td>
          <td>0.300816</td>
          <td>26.012120</td>
          <td>0.090561</td>
          <td>25.952177</td>
          <td>0.075585</td>
          <td>25.716967</td>
          <td>0.100032</td>
          <td>25.667130</td>
          <td>0.180389</td>
          <td>25.465479</td>
          <td>0.325642</td>
          <td>0.070115</td>
          <td>0.038200</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.393663</td>
          <td>0.346947</td>
          <td>26.114309</td>
          <td>0.099050</td>
          <td>25.423996</td>
          <td>0.047319</td>
          <td>25.074588</td>
          <td>0.056721</td>
          <td>24.882772</td>
          <td>0.091531</td>
          <td>24.536874</td>
          <td>0.150685</td>
          <td>0.109835</td>
          <td>0.094265</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.177836</td>
          <td>1.172534</td>
          <td>26.982330</td>
          <td>0.208664</td>
          <td>26.019914</td>
          <td>0.080245</td>
          <td>25.316026</td>
          <td>0.070260</td>
          <td>24.683964</td>
          <td>0.076821</td>
          <td>24.158322</td>
          <td>0.108566</td>
          <td>0.191820</td>
          <td>0.109653</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.563845</td>
          <td>0.808084</td>
          <td>26.516211</td>
          <td>0.140439</td>
          <td>26.369505</td>
          <td>0.109078</td>
          <td>25.926857</td>
          <td>0.120144</td>
          <td>25.669649</td>
          <td>0.180775</td>
          <td>25.896272</td>
          <td>0.454595</td>
          <td>0.198935</td>
          <td>0.121044</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.358259</td>
          <td>0.337395</td>
          <td>26.095159</td>
          <td>0.097403</td>
          <td>26.018281</td>
          <td>0.080129</td>
          <td>25.894356</td>
          <td>0.116795</td>
          <td>25.653226</td>
          <td>0.178276</td>
          <td>25.288693</td>
          <td>0.282555</td>
          <td>0.048308</td>
          <td>0.030128</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.074254</td>
          <td>0.578745</td>
          <td>26.975915</td>
          <td>0.207548</td>
          <td>26.527837</td>
          <td>0.125193</td>
          <td>26.153528</td>
          <td>0.146156</td>
          <td>25.579501</td>
          <td>0.167449</td>
          <td>25.200271</td>
          <td>0.262939</td>
          <td>0.105305</td>
          <td>0.094104</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.756373</td>
          <td>0.506477</td>
          <td>26.771964</td>
          <td>0.200543</td>
          <td>26.149774</td>
          <td>0.105755</td>
          <td>25.017713</td>
          <td>0.063968</td>
          <td>24.754590</td>
          <td>0.096145</td>
          <td>23.973190</td>
          <td>0.109074</td>
          <td>0.009381</td>
          <td>0.009333</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.356150</td>
          <td>0.784257</td>
          <td>27.584933</td>
          <td>0.397242</td>
          <td>26.599067</td>
          <td>0.161012</td>
          <td>26.245689</td>
          <td>0.192300</td>
          <td>25.879590</td>
          <td>0.258829</td>
          <td>25.596141</td>
          <td>0.430446</td>
          <td>0.106074</td>
          <td>0.092584</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.757288</td>
          <td>0.510950</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.370526</td>
          <td>1.195518</td>
          <td>25.998866</td>
          <td>0.152879</td>
          <td>25.118057</td>
          <td>0.133628</td>
          <td>24.278197</td>
          <td>0.143931</td>
          <td>0.074859</td>
          <td>0.046678</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.070856</td>
          <td>1.203143</td>
          <td>28.175279</td>
          <td>0.609730</td>
          <td>27.203469</td>
          <td>0.264440</td>
          <td>26.879927</td>
          <td>0.320622</td>
          <td>25.413809</td>
          <td>0.173792</td>
          <td>25.326943</td>
          <td>0.346319</td>
          <td>0.084409</td>
          <td>0.080464</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.939413</td>
          <td>0.270422</td>
          <td>26.125151</td>
          <td>0.116377</td>
          <td>25.975845</td>
          <td>0.091754</td>
          <td>25.734881</td>
          <td>0.121444</td>
          <td>25.650072</td>
          <td>0.209681</td>
          <td>26.798684</td>
          <td>0.967563</td>
          <td>0.070115</td>
          <td>0.038200</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.531135</td>
          <td>0.437734</td>
          <td>26.430767</td>
          <td>0.154717</td>
          <td>25.469596</td>
          <td>0.060090</td>
          <td>25.098554</td>
          <td>0.071205</td>
          <td>24.738561</td>
          <td>0.098088</td>
          <td>24.748619</td>
          <td>0.219010</td>
          <td>0.109835</td>
          <td>0.094265</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.883746</td>
          <td>0.234832</td>
          <td>26.059343</td>
          <td>0.105302</td>
          <td>25.298962</td>
          <td>0.088657</td>
          <td>24.726166</td>
          <td>0.101073</td>
          <td>24.125075</td>
          <td>0.134294</td>
          <td>0.191820</td>
          <td>0.109653</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.529423</td>
          <td>0.451782</td>
          <td>26.981540</td>
          <td>0.256220</td>
          <td>26.393471</td>
          <td>0.141813</td>
          <td>26.564659</td>
          <td>0.263251</td>
          <td>26.301148</td>
          <td>0.379435</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.198935</td>
          <td>0.121044</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.615141</td>
          <td>0.206195</td>
          <td>26.217472</td>
          <td>0.125495</td>
          <td>26.080639</td>
          <td>0.100064</td>
          <td>25.845355</td>
          <td>0.132932</td>
          <td>25.867620</td>
          <td>0.249896</td>
          <td>24.953778</td>
          <td>0.252352</td>
          <td>0.048308</td>
          <td>0.030128</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.814166</td>
          <td>1.043650</td>
          <td>26.499681</td>
          <td>0.163866</td>
          <td>26.608724</td>
          <td>0.162377</td>
          <td>26.252293</td>
          <td>0.193411</td>
          <td>26.413288</td>
          <td>0.395913</td>
          <td>25.756221</td>
          <td>0.485533</td>
          <td>0.105305</td>
          <td>0.094104</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.295917</td>
          <td>0.676219</td>
          <td>26.969910</td>
          <td>0.206710</td>
          <td>25.946212</td>
          <td>0.075278</td>
          <td>25.201160</td>
          <td>0.063542</td>
          <td>24.668688</td>
          <td>0.075882</td>
          <td>24.046267</td>
          <td>0.098549</td>
          <td>0.009381</td>
          <td>0.009333</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.368720</td>
          <td>0.315471</td>
          <td>26.696831</td>
          <td>0.162620</td>
          <td>26.488849</td>
          <td>0.218779</td>
          <td>25.679085</td>
          <td>0.204122</td>
          <td>27.029115</td>
          <td>1.069473</td>
          <td>0.106074</td>
          <td>0.092584</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.625980</td>
          <td>0.756005</td>
          <td>27.290029</td>
          <td>0.250444</td>
          <td>26.132872</td>
          <td>0.151084</td>
          <td>25.089484</td>
          <td>0.115244</td>
          <td>24.071854</td>
          <td>0.105967</td>
          <td>0.074859</td>
          <td>0.046678</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.130349</td>
          <td>0.549233</td>
          <td>27.162671</td>
          <td>0.232950</td>
          <td>26.856249</td>
          <td>0.286229</td>
          <td>25.456688</td>
          <td>0.163674</td>
          <td>25.841870</td>
          <td>0.470249</td>
          <td>0.084409</td>
          <td>0.080464</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.719674</td>
          <td>0.457246</td>
          <td>26.029767</td>
          <td>0.095327</td>
          <td>25.772928</td>
          <td>0.067223</td>
          <td>25.793178</td>
          <td>0.111596</td>
          <td>25.487624</td>
          <td>0.161106</td>
          <td>25.259156</td>
          <td>0.286898</td>
          <td>0.070115</td>
          <td>0.038200</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.435519</td>
          <td>0.388087</td>
          <td>26.271871</td>
          <td>0.126669</td>
          <td>25.465446</td>
          <td>0.055684</td>
          <td>25.060685</td>
          <td>0.063890</td>
          <td>24.788199</td>
          <td>0.095384</td>
          <td>24.688789</td>
          <td>0.194196</td>
          <td>0.109835</td>
          <td>0.094265</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.811480</td>
          <td>1.792431</td>
          <td>26.817035</td>
          <td>0.223226</td>
          <td>26.032958</td>
          <td>0.103366</td>
          <td>25.158738</td>
          <td>0.078701</td>
          <td>25.174953</td>
          <td>0.149837</td>
          <td>24.393679</td>
          <td>0.169804</td>
          <td>0.191820</td>
          <td>0.109653</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.919093</td>
          <td>0.284478</td>
          <td>26.889901</td>
          <td>0.241629</td>
          <td>26.389465</td>
          <td>0.143943</td>
          <td>26.056543</td>
          <td>0.175483</td>
          <td>25.665309</td>
          <td>0.231500</td>
          <td>26.434442</td>
          <td>0.822790</td>
          <td>0.198935</td>
          <td>0.121044</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.339626</td>
          <td>0.336970</td>
          <td>26.251086</td>
          <td>0.113704</td>
          <td>26.031021</td>
          <td>0.082801</td>
          <td>25.833678</td>
          <td>0.113283</td>
          <td>25.897176</td>
          <td>0.223343</td>
          <td>25.159603</td>
          <td>0.259665</td>
          <td>0.048308</td>
          <td>0.030128</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.792584</td>
          <td>0.506413</td>
          <td>26.759495</td>
          <td>0.191321</td>
          <td>26.421202</td>
          <td>0.128388</td>
          <td>26.288853</td>
          <td>0.185101</td>
          <td>25.643931</td>
          <td>0.198321</td>
          <td>25.198502</td>
          <td>0.294236</td>
          <td>0.105305</td>
          <td>0.094104</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
