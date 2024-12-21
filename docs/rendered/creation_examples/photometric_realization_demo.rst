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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1a4f391000>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>26.732177</td>
          <td>0.450317</td>
          <td>26.805904</td>
          <td>0.179870</td>
          <td>25.987562</td>
          <td>0.077985</td>
          <td>25.196029</td>
          <td>0.063174</td>
          <td>25.034752</td>
          <td>0.104577</td>
          <td>25.100636</td>
          <td>0.242289</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.082163</td>
          <td>1.110329</td>
          <td>28.273150</td>
          <td>0.572860</td>
          <td>27.644425</td>
          <td>0.318772</td>
          <td>26.978914</td>
          <td>0.291393</td>
          <td>26.422184</td>
          <td>0.335465</td>
          <td>25.893617</td>
          <td>0.453688</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.451861</td>
          <td>0.363148</td>
          <td>25.853509</td>
          <td>0.078765</td>
          <td>24.779039</td>
          <td>0.026784</td>
          <td>23.856979</td>
          <td>0.019507</td>
          <td>23.135906</td>
          <td>0.019714</td>
          <td>22.801777</td>
          <td>0.032735</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.725194</td>
          <td>0.895709</td>
          <td>28.087677</td>
          <td>0.500643</td>
          <td>27.406464</td>
          <td>0.263039</td>
          <td>26.507306</td>
          <td>0.197484</td>
          <td>25.832301</td>
          <td>0.207315</td>
          <td>25.406070</td>
          <td>0.310568</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.693248</td>
          <td>0.196019</td>
          <td>25.727921</td>
          <td>0.070502</td>
          <td>25.416568</td>
          <td>0.047008</td>
          <td>24.808043</td>
          <td>0.044767</td>
          <td>24.337673</td>
          <td>0.056528</td>
          <td>23.646732</td>
          <td>0.069212</td>
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
          <td>2.147172</td>
          <td>27.212537</td>
          <td>0.637975</td>
          <td>26.358201</td>
          <td>0.122511</td>
          <td>26.201603</td>
          <td>0.094164</td>
          <td>26.009881</td>
          <td>0.129117</td>
          <td>25.713485</td>
          <td>0.187602</td>
          <td>25.198032</td>
          <td>0.262458</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.747038</td>
          <td>2.415191</td>
          <td>27.046961</td>
          <td>0.220224</td>
          <td>26.608332</td>
          <td>0.134228</td>
          <td>26.985737</td>
          <td>0.293002</td>
          <td>25.969204</td>
          <td>0.232351</td>
          <td>25.772415</td>
          <td>0.413813</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.924953</td>
          <td>0.519582</td>
          <td>27.013521</td>
          <td>0.214173</td>
          <td>27.117299</td>
          <td>0.207056</td>
          <td>26.320971</td>
          <td>0.168672</td>
          <td>25.608445</td>
          <td>0.171625</td>
          <td>25.188965</td>
          <td>0.260520</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.115862</td>
          <td>0.596113</td>
          <td>27.304059</td>
          <td>0.272127</td>
          <td>26.353735</td>
          <td>0.107586</td>
          <td>26.162264</td>
          <td>0.147258</td>
          <td>25.804798</td>
          <td>0.202591</td>
          <td>24.998260</td>
          <td>0.222593</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.071897</td>
          <td>0.577772</td>
          <td>26.504165</td>
          <td>0.138989</td>
          <td>26.196532</td>
          <td>0.093745</td>
          <td>25.719078</td>
          <td>0.100218</td>
          <td>24.995839</td>
          <td>0.101076</td>
          <td>24.823575</td>
          <td>0.192304</td>
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
          <td>0.890625</td>
          <td>26.563942</td>
          <td>0.438677</td>
          <td>26.922357</td>
          <td>0.227302</td>
          <td>26.054825</td>
          <td>0.097295</td>
          <td>25.408215</td>
          <td>0.090280</td>
          <td>24.932582</td>
          <td>0.112306</td>
          <td>24.889550</td>
          <td>0.238099</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.715725</td>
          <td>0.862567</td>
          <td>27.536392</td>
          <td>0.338590</td>
          <td>26.964326</td>
          <td>0.335748</td>
          <td>26.080022</td>
          <td>0.295609</td>
          <td>26.722734</td>
          <td>0.916429</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.750802</td>
          <td>0.511562</td>
          <td>26.064688</td>
          <td>0.111572</td>
          <td>24.745378</td>
          <td>0.031271</td>
          <td>23.852377</td>
          <td>0.023444</td>
          <td>23.179861</td>
          <td>0.024522</td>
          <td>22.857650</td>
          <td>0.041667</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.367511</td>
          <td>0.344790</td>
          <td>27.087592</td>
          <td>0.250689</td>
          <td>26.795239</td>
          <td>0.312438</td>
          <td>25.689987</td>
          <td>0.228675</td>
          <td>25.138158</td>
          <td>0.310657</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.729671</td>
          <td>0.225974</td>
          <td>25.830829</td>
          <td>0.089150</td>
          <td>25.415430</td>
          <td>0.055331</td>
          <td>24.831841</td>
          <td>0.054251</td>
          <td>24.352320</td>
          <td>0.067443</td>
          <td>23.724467</td>
          <td>0.087717</td>
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
          <td>2.147172</td>
          <td>26.793786</td>
          <td>0.527441</td>
          <td>26.582990</td>
          <td>0.174085</td>
          <td>26.012241</td>
          <td>0.095716</td>
          <td>26.274065</td>
          <td>0.194700</td>
          <td>25.880972</td>
          <td>0.256339</td>
          <td>25.795163</td>
          <td>0.494671</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.214990</td>
          <td>2.968731</td>
          <td>26.942592</td>
          <td>0.231967</td>
          <td>26.768563</td>
          <td>0.180931</td>
          <td>26.751480</td>
          <td>0.284195</td>
          <td>26.079473</td>
          <td>0.296553</td>
          <td>25.092127</td>
          <td>0.282150</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.022837</td>
          <td>0.618163</td>
          <td>27.492244</td>
          <td>0.363855</td>
          <td>27.016526</td>
          <td>0.224617</td>
          <td>26.342968</td>
          <td>0.204633</td>
          <td>25.463874</td>
          <td>0.179622</td>
          <td>28.452011</td>
          <td>2.203027</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>30.761900</td>
          <td>3.510119</td>
          <td>27.767114</td>
          <td>0.455908</td>
          <td>26.521746</td>
          <td>0.150527</td>
          <td>26.040697</td>
          <td>0.161412</td>
          <td>25.448932</td>
          <td>0.180575</td>
          <td>26.136623</td>
          <td>0.637712</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.063668</td>
          <td>0.635001</td>
          <td>26.275799</td>
          <td>0.132527</td>
          <td>26.016469</td>
          <td>0.095024</td>
          <td>25.656496</td>
          <td>0.113365</td>
          <td>25.305014</td>
          <td>0.156478</td>
          <td>25.227528</td>
          <td>0.316354</td>
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
          <td>0.890625</td>
          <td>27.002568</td>
          <td>0.549766</td>
          <td>26.778716</td>
          <td>0.175792</td>
          <td>25.911250</td>
          <td>0.072909</td>
          <td>25.367370</td>
          <td>0.073535</td>
          <td>24.995046</td>
          <td>0.101019</td>
          <td>24.785861</td>
          <td>0.186305</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.065807</td>
          <td>0.492959</td>
          <td>28.361728</td>
          <td>0.550997</td>
          <td>27.132163</td>
          <td>0.329726</td>
          <td>27.439677</td>
          <td>0.710902</td>
          <td>25.851860</td>
          <td>0.439985</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.331113</td>
          <td>0.348065</td>
          <td>25.902193</td>
          <td>0.088372</td>
          <td>24.770441</td>
          <td>0.028852</td>
          <td>23.855463</td>
          <td>0.021177</td>
          <td>23.144166</td>
          <td>0.021504</td>
          <td>22.790674</td>
          <td>0.035309</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.161321</td>
          <td>1.289502</td>
          <td>27.543093</td>
          <td>0.394256</td>
          <td>28.928786</td>
          <td>0.954683</td>
          <td>26.854449</td>
          <td>0.326448</td>
          <td>26.349506</td>
          <td>0.387197</td>
          <td>25.535987</td>
          <td>0.422640</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.093301</td>
          <td>0.273087</td>
          <td>25.872943</td>
          <td>0.080225</td>
          <td>25.485197</td>
          <td>0.050033</td>
          <td>24.824634</td>
          <td>0.045500</td>
          <td>24.400299</td>
          <td>0.059844</td>
          <td>23.641696</td>
          <td>0.069006</td>
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
          <td>2.147172</td>
          <td>26.852370</td>
          <td>0.516158</td>
          <td>26.273656</td>
          <td>0.121880</td>
          <td>26.061066</td>
          <td>0.090048</td>
          <td>25.996399</td>
          <td>0.138393</td>
          <td>26.150280</td>
          <td>0.289914</td>
          <td>25.196287</td>
          <td>0.282644</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.015717</td>
          <td>1.076454</td>
          <td>26.627159</td>
          <td>0.156646</td>
          <td>26.554713</td>
          <td>0.130241</td>
          <td>27.003526</td>
          <td>0.301969</td>
          <td>25.708894</td>
          <td>0.189854</td>
          <td>25.486472</td>
          <td>0.336263</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.554557</td>
          <td>1.464154</td>
          <td>26.964865</td>
          <td>0.214167</td>
          <td>26.913801</td>
          <td>0.182825</td>
          <td>26.451644</td>
          <td>0.197922</td>
          <td>26.454350</td>
          <td>0.359470</td>
          <td>25.248983</td>
          <td>0.286704</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.106209</td>
          <td>0.632289</td>
          <td>27.045098</td>
          <td>0.241821</td>
          <td>26.933026</td>
          <td>0.197917</td>
          <td>25.936651</td>
          <td>0.136360</td>
          <td>25.433041</td>
          <td>0.165168</td>
          <td>25.618418</td>
          <td>0.407823</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.060886</td>
          <td>0.586065</td>
          <td>26.454278</td>
          <td>0.137648</td>
          <td>26.116059</td>
          <td>0.090815</td>
          <td>25.676497</td>
          <td>0.100559</td>
          <td>25.075529</td>
          <td>0.112638</td>
          <td>24.960480</td>
          <td>0.224111</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
