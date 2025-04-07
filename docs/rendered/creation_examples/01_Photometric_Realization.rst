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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f6600b33a30>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>28.083594</td>
          <td>1.111246</td>
          <td>26.505144</td>
          <td>0.139107</td>
          <td>26.029911</td>
          <td>0.080956</td>
          <td>25.328652</td>
          <td>0.071050</td>
          <td>24.719046</td>
          <td>0.079238</td>
          <td>24.051064</td>
          <td>0.098842</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.164695</td>
          <td>0.242778</td>
          <td>27.004078</td>
          <td>0.188253</td>
          <td>26.194865</td>
          <td>0.151438</td>
          <td>25.639255</td>
          <td>0.176176</td>
          <td>25.988814</td>
          <td>0.487134</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.189700</td>
          <td>0.485393</td>
          <td>25.846537</td>
          <td>0.112029</td>
          <td>24.975244</td>
          <td>0.099269</td>
          <td>24.169423</td>
          <td>0.109623</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.135758</td>
          <td>2.767970</td>
          <td>27.985786</td>
          <td>0.464128</td>
          <td>27.208393</td>
          <td>0.223408</td>
          <td>26.153651</td>
          <td>0.146172</td>
          <td>25.475305</td>
          <td>0.153184</td>
          <td>26.001300</td>
          <td>0.491662</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.739438</td>
          <td>0.903729</td>
          <td>26.088252</td>
          <td>0.096815</td>
          <td>25.974907</td>
          <td>0.077119</td>
          <td>25.832717</td>
          <td>0.110687</td>
          <td>25.331728</td>
          <td>0.135382</td>
          <td>24.873761</td>
          <td>0.200597</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.383155</td>
          <td>0.125190</td>
          <td>25.420076</td>
          <td>0.047154</td>
          <td>25.068837</td>
          <td>0.056432</td>
          <td>24.801032</td>
          <td>0.085179</td>
          <td>24.694467</td>
          <td>0.172396</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.312122</td>
          <td>2.036661</td>
          <td>26.431267</td>
          <td>0.130514</td>
          <td>26.264887</td>
          <td>0.099539</td>
          <td>25.145793</td>
          <td>0.060421</td>
          <td>24.852310</td>
          <td>0.089112</td>
          <td>24.185496</td>
          <td>0.111171</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.910899</td>
          <td>0.514264</td>
          <td>26.850651</td>
          <td>0.186806</td>
          <td>26.336773</td>
          <td>0.106003</td>
          <td>26.135788</td>
          <td>0.143943</td>
          <td>25.826631</td>
          <td>0.206333</td>
          <td>25.865851</td>
          <td>0.444288</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.415368</td>
          <td>0.352916</td>
          <td>26.209458</td>
          <td>0.107639</td>
          <td>26.184517</td>
          <td>0.092761</td>
          <td>25.817845</td>
          <td>0.109260</td>
          <td>25.702701</td>
          <td>0.185901</td>
          <td>24.922542</td>
          <td>0.208969</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.015766</td>
          <td>0.256125</td>
          <td>26.803845</td>
          <td>0.179557</td>
          <td>26.520781</td>
          <td>0.124429</td>
          <td>26.442080</td>
          <td>0.186920</td>
          <td>25.643812</td>
          <td>0.176858</td>
          <td>26.525448</td>
          <td>0.712797</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>27.717921</td>
          <td>0.968329</td>
          <td>26.583712</td>
          <td>0.171025</td>
          <td>25.994726</td>
          <td>0.092296</td>
          <td>25.119604</td>
          <td>0.069988</td>
          <td>24.632479</td>
          <td>0.086341</td>
          <td>24.093266</td>
          <td>0.121061</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.459268</td>
          <td>0.350997</td>
          <td>26.591868</td>
          <td>0.155056</td>
          <td>26.301848</td>
          <td>0.195242</td>
          <td>25.942805</td>
          <td>0.264474</td>
          <td>25.035647</td>
          <td>0.268478</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.253276</td>
          <td>2.116686</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.844735</td>
          <td>0.438364</td>
          <td>25.735231</td>
          <td>0.122933</td>
          <td>24.980290</td>
          <td>0.119682</td>
          <td>24.172992</td>
          <td>0.132684</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.826932</td>
          <td>0.490052</td>
          <td>27.548857</td>
          <td>0.363027</td>
          <td>26.116422</td>
          <td>0.178411</td>
          <td>25.517000</td>
          <td>0.197919</td>
          <td>25.479076</td>
          <td>0.405920</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.310947</td>
          <td>0.361153</td>
          <td>26.350690</td>
          <td>0.140155</td>
          <td>26.071012</td>
          <td>0.098717</td>
          <td>25.701751</td>
          <td>0.116751</td>
          <td>25.275160</td>
          <td>0.151090</td>
          <td>25.097483</td>
          <td>0.282346</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>27.984365</td>
          <td>1.145088</td>
          <td>26.610957</td>
          <td>0.178262</td>
          <td>25.344937</td>
          <td>0.053076</td>
          <td>25.175023</td>
          <td>0.075130</td>
          <td>24.930139</td>
          <td>0.114425</td>
          <td>24.648547</td>
          <td>0.198805</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.414955</td>
          <td>0.340040</td>
          <td>26.027436</td>
          <td>0.095382</td>
          <td>25.245065</td>
          <td>0.078535</td>
          <td>24.968228</td>
          <td>0.116327</td>
          <td>24.195121</td>
          <td>0.132790</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.842262</td>
          <td>1.767921</td>
          <td>26.491082</td>
          <td>0.159829</td>
          <td>26.186200</td>
          <td>0.110540</td>
          <td>26.337114</td>
          <td>0.203631</td>
          <td>25.907542</td>
          <td>0.259981</td>
          <td>25.423489</td>
          <td>0.370205</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.516554</td>
          <td>0.432069</td>
          <td>26.585387</td>
          <td>0.176037</td>
          <td>26.069386</td>
          <td>0.101683</td>
          <td>25.617887</td>
          <td>0.112046</td>
          <td>25.743822</td>
          <td>0.231193</td>
          <td>25.404478</td>
          <td>0.370998</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.429020</td>
          <td>0.398392</td>
          <td>26.727627</td>
          <td>0.194847</td>
          <td>26.320562</td>
          <td>0.123908</td>
          <td>26.047418</td>
          <td>0.158888</td>
          <td>26.270617</td>
          <td>0.347166</td>
          <td>25.051138</td>
          <td>0.274442</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.629372</td>
          <td>0.416573</td>
          <td>26.323707</td>
          <td>0.118910</td>
          <td>25.902409</td>
          <td>0.072341</td>
          <td>25.214005</td>
          <td>0.064197</td>
          <td>24.630672</td>
          <td>0.073297</td>
          <td>24.210316</td>
          <td>0.113618</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.105587</td>
          <td>0.592101</td>
          <td>27.209964</td>
          <td>0.252180</td>
          <td>26.687169</td>
          <td>0.143804</td>
          <td>26.098185</td>
          <td>0.139491</td>
          <td>25.858328</td>
          <td>0.212068</td>
          <td>25.122821</td>
          <td>0.246985</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.840777</td>
          <td>0.442576</td>
          <td>28.163374</td>
          <td>0.509920</td>
          <td>26.047923</td>
          <td>0.145123</td>
          <td>25.183858</td>
          <td>0.129119</td>
          <td>24.529591</td>
          <td>0.162635</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.059979</td>
          <td>0.244241</td>
          <td>26.215256</td>
          <td>0.193265</td>
          <td>26.122415</td>
          <td>0.323968</td>
          <td>25.435408</td>
          <td>0.391235</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.488251</td>
          <td>0.373925</td>
          <td>26.114568</td>
          <td>0.099194</td>
          <td>25.822951</td>
          <td>0.067515</td>
          <td>25.505308</td>
          <td>0.083174</td>
          <td>25.496399</td>
          <td>0.156192</td>
          <td>25.622170</td>
          <td>0.368917</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.214503</td>
          <td>0.316599</td>
          <td>26.330209</td>
          <td>0.128000</td>
          <td>25.365664</td>
          <td>0.048662</td>
          <td>24.999744</td>
          <td>0.057691</td>
          <td>24.896414</td>
          <td>0.100208</td>
          <td>24.787375</td>
          <td>0.201684</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.713648</td>
          <td>0.168640</td>
          <td>26.191988</td>
          <td>0.094923</td>
          <td>25.110788</td>
          <td>0.059606</td>
          <td>24.848565</td>
          <td>0.090295</td>
          <td>24.350711</td>
          <td>0.130524</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.392612</td>
          <td>0.357500</td>
          <td>26.704241</td>
          <td>0.171960</td>
          <td>26.175994</td>
          <td>0.096686</td>
          <td>26.173938</td>
          <td>0.156369</td>
          <td>25.685706</td>
          <td>0.192058</td>
          <td>25.738150</td>
          <td>0.421292</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.698847</td>
          <td>0.933890</td>
          <td>26.056648</td>
          <td>0.104146</td>
          <td>26.095560</td>
          <td>0.096217</td>
          <td>25.779793</td>
          <td>0.119034</td>
          <td>25.699769</td>
          <td>0.206934</td>
          <td>25.649584</td>
          <td>0.417673</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.166021</td>
          <td>0.631121</td>
          <td>26.827766</td>
          <td>0.189300</td>
          <td>26.519139</td>
          <td>0.129116</td>
          <td>26.583846</td>
          <td>0.218890</td>
          <td>26.205721</td>
          <td>0.292312</td>
          <td>25.316016</td>
          <td>0.299769</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
