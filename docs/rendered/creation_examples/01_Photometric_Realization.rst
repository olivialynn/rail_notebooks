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

    <pzflow.flow.Flow at 0x7f29a3ce5600>



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
    0      23.994413  0.011920  0.009285  
    1      25.391064  0.007621  0.006095  
    2      24.304707  0.077026  0.041644  
    3      25.291103  0.069482  0.052885  
    4      25.096743  0.082620  0.081099  
    ...          ...       ...       ...  
    99995  24.737946  0.076796  0.040889  
    99996  24.224169  0.006449  0.004614  
    99997  25.613836  0.095697  0.090608  
    99998  25.274899  0.107162  0.082091  
    99999  25.699642  0.048109  0.044076  
    
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
          <td>26.836056</td>
          <td>0.486661</td>
          <td>26.661960</td>
          <td>0.159141</td>
          <td>25.976027</td>
          <td>0.077195</td>
          <td>25.198598</td>
          <td>0.063318</td>
          <td>24.593369</td>
          <td>0.070908</td>
          <td>24.150725</td>
          <td>0.107848</td>
          <td>0.011920</td>
          <td>0.009285</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.997248</td>
          <td>0.547617</td>
          <td>27.376590</td>
          <td>0.288609</td>
          <td>26.592759</td>
          <td>0.132433</td>
          <td>26.336682</td>
          <td>0.170943</td>
          <td>26.093299</td>
          <td>0.257358</td>
          <td>25.153666</td>
          <td>0.253093</td>
          <td>0.007621</td>
          <td>0.006095</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.311137</td>
          <td>2.035826</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.725597</td>
          <td>0.339991</td>
          <td>25.896342</td>
          <td>0.116997</td>
          <td>25.214511</td>
          <td>0.122314</td>
          <td>24.550072</td>
          <td>0.152401</td>
          <td>0.077026</td>
          <td>0.041644</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.795160</td>
          <td>0.401587</td>
          <td>27.676332</td>
          <td>0.326973</td>
          <td>25.979168</td>
          <td>0.125726</td>
          <td>25.238086</td>
          <td>0.124842</td>
          <td>25.723684</td>
          <td>0.398613</td>
          <td>0.069482</td>
          <td>0.052885</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.325343</td>
          <td>0.328717</td>
          <td>26.409021</td>
          <td>0.128026</td>
          <td>26.024043</td>
          <td>0.080538</td>
          <td>25.581516</td>
          <td>0.088815</td>
          <td>25.405524</td>
          <td>0.144274</td>
          <td>25.103084</td>
          <td>0.242778</td>
          <td>0.082620</td>
          <td>0.081099</td>
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
          <td>26.852445</td>
          <td>0.492603</td>
          <td>26.493601</td>
          <td>0.137730</td>
          <td>25.366470</td>
          <td>0.044963</td>
          <td>25.185105</td>
          <td>0.062565</td>
          <td>24.862278</td>
          <td>0.089896</td>
          <td>24.603931</td>
          <td>0.159592</td>
          <td>0.076796</td>
          <td>0.040889</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.070315</td>
          <td>0.577120</td>
          <td>26.625635</td>
          <td>0.154274</td>
          <td>26.109229</td>
          <td>0.086818</td>
          <td>25.190107</td>
          <td>0.062843</td>
          <td>24.713476</td>
          <td>0.078850</td>
          <td>24.198772</td>
          <td>0.112466</td>
          <td>0.006449</td>
          <td>0.004614</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.695058</td>
          <td>0.878891</td>
          <td>26.919323</td>
          <td>0.197929</td>
          <td>26.200185</td>
          <td>0.094047</td>
          <td>26.093002</td>
          <td>0.138734</td>
          <td>26.111213</td>
          <td>0.261160</td>
          <td>26.428937</td>
          <td>0.667423</td>
          <td>0.095697</td>
          <td>0.090608</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.595980</td>
          <td>0.406028</td>
          <td>26.015588</td>
          <td>0.090837</td>
          <td>26.136742</td>
          <td>0.088946</td>
          <td>25.828667</td>
          <td>0.110297</td>
          <td>25.785922</td>
          <td>0.199406</td>
          <td>25.205399</td>
          <td>0.264043</td>
          <td>0.107162</td>
          <td>0.082091</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.896608</td>
          <td>0.508900</td>
          <td>26.735954</td>
          <td>0.169503</td>
          <td>26.529018</td>
          <td>0.125321</td>
          <td>26.395290</td>
          <td>0.179664</td>
          <td>26.016804</td>
          <td>0.241674</td>
          <td>26.220821</td>
          <td>0.576800</td>
          <td>0.048109</td>
          <td>0.044076</td>
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
          <td>27.149858</td>
          <td>0.670112</td>
          <td>27.081276</td>
          <td>0.259174</td>
          <td>26.121831</td>
          <td>0.103212</td>
          <td>25.276210</td>
          <td>0.080402</td>
          <td>24.803639</td>
          <td>0.100377</td>
          <td>24.088622</td>
          <td>0.120617</td>
          <td>0.011920</td>
          <td>0.009285</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.546765</td>
          <td>0.870734</td>
          <td>27.067598</td>
          <td>0.256240</td>
          <td>26.536782</td>
          <td>0.147891</td>
          <td>26.193633</td>
          <td>0.178172</td>
          <td>25.760476</td>
          <td>0.227595</td>
          <td>25.751110</td>
          <td>0.470139</td>
          <td>0.007621</td>
          <td>0.006095</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.187191</td>
          <td>2.053225</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.815500</td>
          <td>0.425238</td>
          <td>25.987869</td>
          <td>0.151413</td>
          <td>24.801870</td>
          <td>0.101474</td>
          <td>24.385594</td>
          <td>0.157786</td>
          <td>0.077026</td>
          <td>0.041644</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.894708</td>
          <td>0.494282</td>
          <td>27.095964</td>
          <td>0.239889</td>
          <td>26.477584</td>
          <td>0.228938</td>
          <td>25.753760</td>
          <td>0.229046</td>
          <td>26.445507</td>
          <td>0.774651</td>
          <td>0.069482</td>
          <td>0.052885</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.159717</td>
          <td>0.325437</td>
          <td>26.158344</td>
          <td>0.121031</td>
          <td>25.909089</td>
          <td>0.087536</td>
          <td>25.477397</td>
          <td>0.098170</td>
          <td>25.298416</td>
          <td>0.157463</td>
          <td>25.633493</td>
          <td>0.438791</td>
          <td>0.082620</td>
          <td>0.081099</td>
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
          <td>27.748591</td>
          <td>0.993076</td>
          <td>26.512266</td>
          <td>0.162726</td>
          <td>25.467277</td>
          <td>0.058662</td>
          <td>24.973894</td>
          <td>0.062334</td>
          <td>24.757479</td>
          <td>0.097588</td>
          <td>24.590206</td>
          <td>0.187723</td>
          <td>0.076796</td>
          <td>0.040889</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.130927</td>
          <td>1.230440</td>
          <td>26.327818</td>
          <td>0.137390</td>
          <td>26.087790</td>
          <td>0.100153</td>
          <td>25.148780</td>
          <td>0.071823</td>
          <td>24.802470</td>
          <td>0.100246</td>
          <td>24.299839</td>
          <td>0.144737</td>
          <td>0.006449</td>
          <td>0.004614</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.452543</td>
          <td>0.833034</td>
          <td>26.936441</td>
          <td>0.235657</td>
          <td>26.365061</td>
          <td>0.131169</td>
          <td>26.184015</td>
          <td>0.181833</td>
          <td>25.956189</td>
          <td>0.274521</td>
          <td>26.057967</td>
          <td>0.602130</td>
          <td>0.095697</td>
          <td>0.090608</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.350937</td>
          <td>0.380155</td>
          <td>26.472439</td>
          <td>0.159708</td>
          <td>26.088563</td>
          <td>0.103257</td>
          <td>25.754414</td>
          <td>0.125982</td>
          <td>25.438540</td>
          <td>0.178749</td>
          <td>25.688087</td>
          <td>0.460339</td>
          <td>0.107162</td>
          <td>0.082091</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.392810</td>
          <td>0.386673</td>
          <td>26.789311</td>
          <td>0.204694</td>
          <td>26.604354</td>
          <td>0.157789</td>
          <td>26.278552</td>
          <td>0.192777</td>
          <td>26.639036</td>
          <td>0.459817</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.048109</td>
          <td>0.044076</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.678825</td>
          <td>0.161660</td>
          <td>26.020653</td>
          <td>0.080422</td>
          <td>25.201673</td>
          <td>0.063595</td>
          <td>24.884496</td>
          <td>0.091812</td>
          <td>23.877867</td>
          <td>0.085024</td>
          <td>0.011920</td>
          <td>0.009285</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.204062</td>
          <td>0.250900</td>
          <td>26.438124</td>
          <td>0.115877</td>
          <td>25.977508</td>
          <td>0.125629</td>
          <td>26.159220</td>
          <td>0.271755</td>
          <td>25.645826</td>
          <td>0.375514</td>
          <td>0.007621</td>
          <td>0.006095</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.905704</td>
          <td>0.408470</td>
          <td>26.011499</td>
          <td>0.135965</td>
          <td>25.244012</td>
          <td>0.131661</td>
          <td>24.452363</td>
          <td>0.147241</td>
          <td>0.077026</td>
          <td>0.041644</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.765317</td>
          <td>0.161160</td>
          <td>26.256973</td>
          <td>0.167881</td>
          <td>25.041069</td>
          <td>0.110408</td>
          <td>24.949763</td>
          <td>0.224338</td>
          <td>0.069482</td>
          <td>0.052885</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.716623</td>
          <td>0.468166</td>
          <td>26.026621</td>
          <td>0.098673</td>
          <td>25.981323</td>
          <td>0.084339</td>
          <td>25.678974</td>
          <td>0.105551</td>
          <td>25.765367</td>
          <td>0.212200</td>
          <td>24.973457</td>
          <td>0.236582</td>
          <td>0.082620</td>
          <td>0.081099</td>
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
          <td>27.972402</td>
          <td>1.065338</td>
          <td>26.448328</td>
          <td>0.138048</td>
          <td>25.451097</td>
          <td>0.050896</td>
          <td>25.143319</td>
          <td>0.063443</td>
          <td>24.818327</td>
          <td>0.090759</td>
          <td>24.486240</td>
          <td>0.151492</td>
          <td>0.076796</td>
          <td>0.040889</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.194115</td>
          <td>0.629983</td>
          <td>26.835772</td>
          <td>0.184538</td>
          <td>26.029018</td>
          <td>0.080927</td>
          <td>25.139432</td>
          <td>0.060108</td>
          <td>24.839154</td>
          <td>0.088124</td>
          <td>24.092460</td>
          <td>0.102537</td>
          <td>0.006449</td>
          <td>0.004614</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.528415</td>
          <td>0.411744</td>
          <td>26.906094</td>
          <td>0.213830</td>
          <td>26.153495</td>
          <td>0.100305</td>
          <td>26.654788</td>
          <td>0.247819</td>
          <td>26.026779</td>
          <td>0.268824</td>
          <td>25.778224</td>
          <td>0.456699</td>
          <td>0.095697</td>
          <td>0.090608</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.472677</td>
          <td>0.395549</td>
          <td>26.074439</td>
          <td>0.105295</td>
          <td>25.945399</td>
          <td>0.083878</td>
          <td>25.980868</td>
          <td>0.140905</td>
          <td>25.828735</td>
          <td>0.229278</td>
          <td>25.518107</td>
          <td>0.375620</td>
          <td>0.107162</td>
          <td>0.082091</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.416912</td>
          <td>0.744489</td>
          <td>27.074699</td>
          <td>0.230620</td>
          <td>26.828372</td>
          <td>0.166643</td>
          <td>26.266777</td>
          <td>0.165729</td>
          <td>25.804892</td>
          <td>0.208100</td>
          <td>25.352709</td>
          <td>0.305586</td>
          <td>0.048109</td>
          <td>0.044076</td>
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
