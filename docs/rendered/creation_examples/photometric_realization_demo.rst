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

    <pzflow.flow.Flow at 0x7f54a6b564d0>



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
          <td>26.611192</td>
          <td>0.410790</td>
          <td>26.608365</td>
          <td>0.152008</td>
          <td>25.969601</td>
          <td>0.076758</td>
          <td>25.268099</td>
          <td>0.067341</td>
          <td>24.798776</td>
          <td>0.085010</td>
          <td>24.881653</td>
          <td>0.201930</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.340729</td>
          <td>2.060930</td>
          <td>27.850698</td>
          <td>0.419050</td>
          <td>27.668179</td>
          <td>0.324860</td>
          <td>27.597365</td>
          <td>0.471593</td>
          <td>28.117323</td>
          <td>1.088083</td>
          <td>28.615846</td>
          <td>2.167605</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.395072</td>
          <td>0.347332</td>
          <td>25.903092</td>
          <td>0.082282</td>
          <td>24.738883</td>
          <td>0.025863</td>
          <td>23.877565</td>
          <td>0.019850</td>
          <td>23.159976</td>
          <td>0.020120</td>
          <td>22.871262</td>
          <td>0.034803</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.776130</td>
          <td>0.395744</td>
          <td>27.489689</td>
          <td>0.281474</td>
          <td>26.747111</td>
          <td>0.241159</td>
          <td>26.557021</td>
          <td>0.372945</td>
          <td>25.247212</td>
          <td>0.273198</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.388249</td>
          <td>0.345472</td>
          <td>25.852606</td>
          <td>0.078703</td>
          <td>25.419354</td>
          <td>0.047124</td>
          <td>24.887126</td>
          <td>0.048023</td>
          <td>24.354457</td>
          <td>0.057377</td>
          <td>23.727192</td>
          <td>0.074317</td>
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
          <td>26.860775</td>
          <td>0.495645</td>
          <td>26.581333</td>
          <td>0.148525</td>
          <td>26.026031</td>
          <td>0.080679</td>
          <td>25.872816</td>
          <td>0.114625</td>
          <td>26.282812</td>
          <td>0.300156</td>
          <td>25.446399</td>
          <td>0.320733</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.698267</td>
          <td>0.164149</td>
          <td>27.152362</td>
          <td>0.213217</td>
          <td>26.284058</td>
          <td>0.163447</td>
          <td>26.263675</td>
          <td>0.295569</td>
          <td>25.465923</td>
          <td>0.325757</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.340984</td>
          <td>0.332817</td>
          <td>27.145186</td>
          <td>0.238903</td>
          <td>26.857099</td>
          <td>0.166183</td>
          <td>26.438527</td>
          <td>0.186360</td>
          <td>26.229904</td>
          <td>0.287622</td>
          <td>25.809979</td>
          <td>0.425853</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.501852</td>
          <td>0.319130</td>
          <td>26.468993</td>
          <td>0.118955</td>
          <td>26.017511</td>
          <td>0.129973</td>
          <td>25.697053</td>
          <td>0.185016</td>
          <td>25.738178</td>
          <td>0.403085</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.790001</td>
          <td>0.470269</td>
          <td>26.737337</td>
          <td>0.169702</td>
          <td>26.090088</td>
          <td>0.085366</td>
          <td>25.658156</td>
          <td>0.095004</td>
          <td>25.265896</td>
          <td>0.127889</td>
          <td>25.111351</td>
          <td>0.244438</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.731688</td>
          <td>0.193825</td>
          <td>26.076053</td>
          <td>0.099122</td>
          <td>25.286671</td>
          <td>0.081118</td>
          <td>25.068431</td>
          <td>0.126381</td>
          <td>24.731861</td>
          <td>0.208845</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.169901</td>
          <td>0.597630</td>
          <td>27.547578</td>
          <td>0.341596</td>
          <td>27.491248</td>
          <td>0.502529</td>
          <td>26.623157</td>
          <td>0.451666</td>
          <td>26.302730</td>
          <td>0.697207</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.542683</td>
          <td>0.879138</td>
          <td>25.991533</td>
          <td>0.104678</td>
          <td>24.759457</td>
          <td>0.031660</td>
          <td>23.840650</td>
          <td>0.023209</td>
          <td>23.128536</td>
          <td>0.023460</td>
          <td>22.903276</td>
          <td>0.043386</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.539950</td>
          <td>0.394427</td>
          <td>28.363427</td>
          <td>0.662306</td>
          <td>26.696811</td>
          <td>0.288670</td>
          <td>26.080870</td>
          <td>0.314430</td>
          <td>25.050257</td>
          <td>0.289461</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.716254</td>
          <td>0.491730</td>
          <td>25.770325</td>
          <td>0.084534</td>
          <td>25.480764</td>
          <td>0.058632</td>
          <td>24.860588</td>
          <td>0.055652</td>
          <td>24.362452</td>
          <td>0.068050</td>
          <td>23.753688</td>
          <td>0.090000</td>
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
          <td>27.510971</td>
          <td>0.861022</td>
          <td>26.682543</td>
          <td>0.189379</td>
          <td>26.056801</td>
          <td>0.099528</td>
          <td>26.131120</td>
          <td>0.172526</td>
          <td>25.567043</td>
          <td>0.197514</td>
          <td>25.713916</td>
          <td>0.465655</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.556360</td>
          <td>0.877968</td>
          <td>26.800190</td>
          <td>0.206035</td>
          <td>26.766759</td>
          <td>0.180655</td>
          <td>26.705430</td>
          <td>0.273772</td>
          <td>25.696433</td>
          <td>0.216615</td>
          <td>25.060917</td>
          <td>0.275094</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.205168</td>
          <td>0.335165</td>
          <td>26.875004</td>
          <td>0.220933</td>
          <td>27.034718</td>
          <td>0.228035</td>
          <td>26.431445</td>
          <td>0.220329</td>
          <td>26.184658</td>
          <td>0.325135</td>
          <td>27.954609</td>
          <td>1.785569</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.641706</td>
          <td>0.939762</td>
          <td>27.116656</td>
          <td>0.273804</td>
          <td>26.636915</td>
          <td>0.166105</td>
          <td>25.851630</td>
          <td>0.137229</td>
          <td>25.780648</td>
          <td>0.238343</td>
          <td>26.074936</td>
          <td>0.610751</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>30.889111</td>
          <td>3.614611</td>
          <td>26.627134</td>
          <td>0.179002</td>
          <td>26.059136</td>
          <td>0.098647</td>
          <td>25.779669</td>
          <td>0.126174</td>
          <td>25.122346</td>
          <td>0.133732</td>
          <td>24.392133</td>
          <td>0.158232</td>
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
          <td>27.540570</td>
          <td>0.795990</td>
          <td>26.235661</td>
          <td>0.110140</td>
          <td>25.966472</td>
          <td>0.076556</td>
          <td>25.270178</td>
          <td>0.067474</td>
          <td>25.095886</td>
          <td>0.110328</td>
          <td>24.737724</td>
          <td>0.178868</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.045379</td>
          <td>1.815920</td>
          <td>27.865654</td>
          <td>0.424159</td>
          <td>27.854331</td>
          <td>0.376421</td>
          <td>27.087739</td>
          <td>0.318273</td>
          <td>26.587292</td>
          <td>0.382146</td>
          <td>25.476911</td>
          <td>0.328906</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.437071</td>
          <td>0.378113</td>
          <td>25.965904</td>
          <td>0.093455</td>
          <td>24.828922</td>
          <td>0.030370</td>
          <td>23.873497</td>
          <td>0.021506</td>
          <td>23.129539</td>
          <td>0.021237</td>
          <td>22.837144</td>
          <td>0.036789</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.542570</td>
          <td>1.567919</td>
          <td>27.989278</td>
          <td>0.550348</td>
          <td>27.850123</td>
          <td>0.456017</td>
          <td>26.957707</td>
          <td>0.354197</td>
          <td>26.339822</td>
          <td>0.384304</td>
          <td>25.179639</td>
          <td>0.320055</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.785271</td>
          <td>0.469002</td>
          <td>25.628633</td>
          <td>0.064657</td>
          <td>25.382372</td>
          <td>0.045668</td>
          <td>24.817831</td>
          <td>0.045226</td>
          <td>24.412606</td>
          <td>0.060501</td>
          <td>23.724153</td>
          <td>0.074228</td>
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
          <td>27.150004</td>
          <td>0.638418</td>
          <td>26.428581</td>
          <td>0.139347</td>
          <td>26.024806</td>
          <td>0.087221</td>
          <td>26.057304</td>
          <td>0.145845</td>
          <td>26.132474</td>
          <td>0.285771</td>
          <td>25.421126</td>
          <td>0.338391</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.046991</td>
          <td>0.572903</td>
          <td>27.163749</td>
          <td>0.245866</td>
          <td>27.231735</td>
          <td>0.231350</td>
          <td>26.560486</td>
          <td>0.209924</td>
          <td>25.958636</td>
          <td>0.233925</td>
          <td>25.640718</td>
          <td>0.379502</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.209914</td>
          <td>0.654357</td>
          <td>27.109050</td>
          <td>0.241375</td>
          <td>26.773877</td>
          <td>0.162323</td>
          <td>26.340019</td>
          <td>0.180127</td>
          <td>25.984873</td>
          <td>0.246431</td>
          <td>25.656944</td>
          <td>0.395847</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.730553</td>
          <td>2.484046</td>
          <td>27.164972</td>
          <td>0.266795</td>
          <td>26.436760</td>
          <td>0.129548</td>
          <td>26.036225</td>
          <td>0.148566</td>
          <td>25.536670</td>
          <td>0.180374</td>
          <td>25.816437</td>
          <td>0.473761</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.401034</td>
          <td>0.357570</td>
          <td>26.375688</td>
          <td>0.128615</td>
          <td>26.178003</td>
          <td>0.095893</td>
          <td>25.827525</td>
          <td>0.114741</td>
          <td>25.469517</td>
          <td>0.158311</td>
          <td>25.190142</td>
          <td>0.270735</td>
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
